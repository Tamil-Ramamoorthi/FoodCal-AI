import hashlib
import hmac
import json
import os
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from io import BytesIO
from typing import Annotated

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .database.history_store import HistoryStore
from .services.nutrition import (
    MAX_GRAMS,
    REFERENCE_PATH as NUTRITION_REFERENCE_FILE,
    NutritionDataError,
    NutritionReference,
)
from .models.classifier import (
    build_eval_transform,
    load_classifier,
    preprocess_image as shared_preprocess_image,
)


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    ROOT
    / "models"
    / "efficientnet_b0"
    / "food_efficientnet_b0_best.pth"
)

# Written by training/evaluate_efficientnet.py and
# training/verify_dataset.py respectively.
EVALUATION_REPORT_PATH = (
    ROOT / "models" / "efficientnet_b0" / "evaluation_report.json"
)
DATASET_REPORT_PATH = (
    ROOT / "data" / "metadata" / "verification_report.json"
)

# Written by evaluation/evaluate_real_world.py. FOODCAL_REAL_WORLD_REPORT
# lets tests point at a temporary report.
REAL_WORLD_REPORT_PATH = Path(
    os.environ.get("FOODCAL_REAL_WORLD_REPORT")
    or ROOT / "reports" / "real_world" / "real_world_evaluation.json"
).resolve()

# Prediction history (SQLite) and thumbnails. FOODCAL_DATA_DIR lets tests
# use a temporary directory instead of the real history.
DATA_DIR = Path(
    os.environ.get("FOODCAL_DATA_DIR") or ROOT / "storage"
).resolve()

DEFAULT_CORS_ORIGINS = [
    # The deployed frontend (Vercel), which reaches this backend through
    # the Cloudflare Tunnel. See docs/DEPLOYMENT.md.
    "https://food-cal-ai-sigma.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
# Browsers send the Origin header without a trailing slash, so a pasted
# "https://example.com/" would otherwise never match.
CORS_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.environ.get("FOODCAL_CORS_ORIGINS", "").split(",")
    if origin.strip()
] or DEFAULT_CORS_ORIGINS

THUMBNAIL_SIZE = 256

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Feedback, deletion and clearing change or remove saved history. A
# request made directly on this computer (the local dev frontend) may do
# so as before. A request that arrived through a proxy such as the
# Cloudflare Tunnel needs this token in the X-FoodCal-Admin-Token header,
# and is refused when no token is set. There are no user accounts.
ADMIN_TOKEN = os.environ.get("FOODCAL_ADMIN_TOKEN", "").strip()
ADMIN_TOKEN_HEADER = "X-FoodCal-Admin-Token"

# Headers that cloudflared and other reverse proxies add to a request.
PROXY_HEADERS = (
    "cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-forwarded-host",
    "x-real-ip", "forwarded",
)
LOOPBACK_HOSTS = {"127.0.0.1", "::1"}

# Predictions per client per minute through a proxy. Each saved prediction
# adds a database row and a thumbnail on this computer's disk.
REMOTE_PREDICTIONS_PER_MINUTE = 20

# Top-1 confidence (percent) below which a prediction is flagged as
# uncertain. 50 was chosen on the validation split by
# evaluation/diagnose_reliability.py: validation predictions below 50%
# confidence were right 48.7% of the time (test split: 47.3%), i.e. more
# likely wrong than right. It says nothing about foods outside the 34
# classes, which can still score high. Override with
# FOODCAL_LOW_CONFIDENCE_THRESHOLD (0-100).
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 50.0


def read_low_confidence_threshold():
    raw = os.environ.get("FOODCAL_LOW_CONFIDENCE_THRESHOLD", "").strip()
    if not raw:
        return DEFAULT_LOW_CONFIDENCE_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not 0 <= value <= 100:
        raise ValueError(
            "FOODCAL_LOW_CONFIDENCE_THRESHOLD must be a number from 0 to 100 "
            f"(a percentage), got {raw!r}."
        )
    return value


LOW_CONFIDENCE_THRESHOLD = read_low_confidence_threshold()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. LOAD TRAINED EFFICIENTNET-B0 MODEL
# =========================================================

classifier = load_classifier(CHECKPOINT_PATH, DEVICE)

model = classifier.model
CLASS_NAMES = classifier.class_names
IMAGE_SIZE = classifier.image_size
CHECKPOINT_EPOCH = classifier.epoch
CHECKPOINT_VAL_ACCURACY = classifier.val_accuracy

# Lets reports be matched to the exact checkpoint being served.
with open(CHECKPOINT_PATH, "rb") as checkpoint_file:
    CHECKPOINT_SHA256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()


# =========================================================
# 3. IMAGE PREPROCESSING
# =========================================================

# Shared with evaluation/evaluate_real_world.py; see
# backend/app/models/classifier.py for why it is built this way.
eval_transform = build_eval_transform(IMAGE_SIZE)


def preprocess_image(image):
    return shared_preprocess_image(image, IMAGE_SIZE, eval_transform)


def make_thumbnail(image):
    # Re-encoded from decoded pixels, so no uploaded bytes are kept.
    thumbnail = ImageOps.exif_transpose(image).convert("RGB")
    thumbnail.thumbnail(
        (THUMBNAIL_SIZE, THUMBNAIL_SIZE),
        Image.Resampling.LANCZOS,
    )
    buffer = BytesIO()
    thumbnail.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


# =========================================================
# 3b. PREDICTION HISTORY STORAGE
# =========================================================

history_store = HistoryStore(DATA_DIR)

try:
    history_store.initialize()
    STORAGE_ERROR = None
except (OSError, sqlite3.Error, RuntimeError) as error:
    # Predictions keep working; history endpoints report the problem. The
    # details (which can include local paths) only go to the console.
    STORAGE_ERROR = (
        "History storage is unavailable. "
        "The backend console shows the cause."
    )
    print(f"WARNING: history storage is unavailable: {error} (data directory: {DATA_DIR})")


def require_storage():
    if STORAGE_ERROR:
        raise HTTPException(status_code=503, detail=STORAGE_ERROR)


# =========================================================
# 3b-2. ACCESS CONTROL AND LIMITS FOR PROXIED REQUESTS
# =========================================================

def is_proxied(request):
    # uvicorn trusts X-Forwarded-For from 127.0.0.1 (where cloudflared
    # connects from), so a tunnelled request has the visitor's address as
    # its client and keeps the proxy headers.
    client = request.client.host if request.client else ""
    return client not in LOOPBACK_HOSTS or any(
        header in request.headers for header in PROXY_HEADERS
    )


def require_admin(request: Request):
    if not is_proxied(request):
        return
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail=(
                "Changing or deleting saved history is disabled for remote "
                "access. Set FOODCAL_ADMIN_TOKEN on the backend to enable it."
            ),
        )
    supplied = request.headers.get(ADMIN_TOKEN_HEADER, "")
    if not hmac.compare_digest(supplied.encode(), ADMIN_TOKEN.encode()):
        raise HTTPException(
            status_code=403,
            detail=(
                "A valid admin token is needed to change or delete saved "
                "history. Enter it in Settings."
            ),
        )


_recent_remote_predictions = defaultdict(deque)


# async, so it runs on the event loop and never concurrently with itself.
async def limit_remote_predictions(request: Request):
    if not is_proxied(request):
        return
    # Cloudflare sets CF-Connecting-IP itself, so a visitor cannot forge it.
    client = request.headers.get("cf-connecting-ip") or (
        request.client.host if request.client else "unknown"
    )
    now = time.monotonic()
    for key in list(_recent_remote_predictions):
        times = _recent_remote_predictions[key]
        while times and now - times[0] >= 60:
            times.popleft()
        if not times:
            del _recent_remote_predictions[key]
    times = _recent_remote_predictions[client]
    if len(times) >= REMOTE_PREDICTIONS_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="Too many predictions from this address. Try again in a minute.",
            headers={"Retry-After": "60"},
        )
    times.append(now)


def to_api_record(record):
    record = dict(record)
    available = record.pop("thumbnail_available")
    record["thumbnail_url"] = (
        f"/api/history/{record['id']}/thumbnail" if available else None
    )
    return record


PredictionId = Annotated[str, PathParam(pattern=r"^[0-9a-f]{32}$")]


# =========================================================
# 3c. NUTRITION REFERENCE (read-only, separate from history)
# =========================================================

# FOODCAL_NUTRITION_REFERENCE lets tests point at a temporary reference file.
NUTRITION_REFERENCE_PATH = Path(
    os.environ.get("FOODCAL_NUTRITION_REFERENCE") or NUTRITION_REFERENCE_FILE
).resolve()

try:
    nutrition_reference = NutritionReference.load(CLASS_NAMES, NUTRITION_REFERENCE_PATH)
    NUTRITION_ERROR = None
except NutritionDataError as error:
    # Predictions keep working; nutrition endpoints report the problem.
    nutrition_reference = None
    NUTRITION_ERROR = (
        "Nutrition data is unavailable. "
        "The backend console shows the cause."
    )
    print(f"WARNING: nutrition data is unavailable: {error}")


def require_nutrition():
    if NUTRITION_ERROR:
        raise HTTPException(status_code=503, detail=NUTRITION_ERROR)


class FeedbackRequest(BaseModel):
    actual_food: str = Field(min_length=1, max_length=100)


# =========================================================
# 4. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="FoodCal AI API",
    description=(
        "Food image classification using the trained "
        "EfficientNet-B0 model."
    ),
    version="1.0.0",
)


# =========================================================
# 5. CORS CONFIGURATION
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(sqlite3.Error)
async def storage_error_handler(request: Request, error: sqlite3.Error):
    return JSONResponse(
        status_code=500,
        content={"detail": "The history database could not be accessed."},
    )


# =========================================================
# 6. HOME ENDPOINT
# =========================================================

@app.get("/")
def home():
    return {
        "message": "FoodCal AI API is running",
        "model": "EfficientNet-B0",
        "classes": len(CLASS_NAMES),
        "device": str(DEVICE),
        "history_storage": "unavailable" if STORAGE_ERROR else "ok",
        # Lets the frontend tell an outdated or misconfigured backend apart
        # from a missing food class.
        "nutrition_reference": "unavailable" if NUTRITION_ERROR else "ok",
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
    }


# =========================================================
# 7. GET AVAILABLE FOOD CLASSES
# =========================================================

@app.get("/api/classes")
def get_classes():
    return {
        "classes": CLASS_NAMES,
        "count": len(CLASS_NAMES),
    }


# =========================================================
# 7b. RECORDED OFFLINE EVALUATION REPORT
# =========================================================

@app.get("/api/evaluation")
def get_evaluation():
    # Serves the metrics recorded by evaluate_efficientnet.py as-is;
    # nothing is recomputed or estimated here.
    if not EVALUATION_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No offline evaluation report found. Run "
                "training/evaluate_efficientnet.py to create one."
            ),
        )

    try:
        report = json.loads(
            EVALUATION_REPORT_PATH.read_text(encoding="utf-8-sig")
        )
        per_class = [
            {
                "food": food,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
                "support": metrics["support"],
            }
            for food, metrics in report["classification_report"].items()
        ]
        summary = {
            key: report[key]
            for key in (
                "model_name",
                "test_images",
                "num_classes",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            )
        }
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(
            status_code=500,
            detail="The offline evaluation report could not be read.",
        )

    dataset_splits = None
    if DATASET_REPORT_PATH.exists():
        try:
            dataset_splits = json.loads(
                DATASET_REPORT_PATH.read_text(encoding="utf-8-sig")
            ).get("split_distribution")
        except (OSError, ValueError):
            dataset_splits = None

    modified = datetime.fromtimestamp(
        EVALUATION_REPORT_PATH.stat().st_mtime
    )

    return {
        **summary,
        "per_class": per_class,
        "classes_match_model": (
            sorted(item["food"] for item in per_class)
            == sorted(CLASS_NAMES)
        ),
        "report_file": "models/efficientnet_b0/evaluation_report.json",
        "report_modified": modified.isoformat(timespec="seconds"),
        "dataset_splits": dataset_splits,
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "checkpoint_val_accuracy": CHECKPOINT_VAL_ACCURACY,
    }


# =========================================================
# 7c. RECORDED REAL-WORLD EVALUATION REPORT
# =========================================================

@app.get("/api/evaluation/real-world")
def get_real_world_evaluation():
    # Serves the report written by evaluation/evaluate_real_world.py as-is;
    # nothing is recomputed or estimated here. Kept separate from the
    # offline test-set report at /api/evaluation.
    if not REAL_WORLD_REPORT_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "No real-world evaluation report found. Run "
                "python -m evaluation.evaluate_real_world to create one."
            ),
        )

    unreadable = HTTPException(
        status_code=500,
        detail="The real-world evaluation report could not be read.",
    )
    try:
        report = json.loads(
            REAL_WORLD_REPORT_PATH.read_text(encoding="utf-8-sig")
        )
        if (
            report.get("report_type") != "real_world_evaluation"
            or report.get("schema_version") != 1
        ):
            raise unreadable
        class_names = [item["name"] for item in report["class_mapping"]]
        report_checkpoint_sha256 = report["checkpoint"]["sha256"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise unreadable

    # Per-image predictions stay in the JSON file; the dashboard only
    # needs the aggregated results.
    report.pop("predictions", None)
    modified = datetime.fromtimestamp(REAL_WORLD_REPORT_PATH.stat().st_mtime)

    return {
        **report,
        "report_file": (
            REAL_WORLD_REPORT_PATH.relative_to(ROOT).as_posix()
            if REAL_WORLD_REPORT_PATH.is_relative_to(ROOT)
            else REAL_WORLD_REPORT_PATH.name
        ),
        "report_modified": modified.isoformat(timespec="seconds"),
        "checkpoint_matches_deployed": (
            report_checkpoint_sha256 == CHECKPOINT_SHA256
        ),
        "classes_match_model": class_names == CLASS_NAMES,
    }


# =========================================================
# 8. FOOD IMAGE PREDICTION ENDPOINT
# =========================================================

@app.post("/api/predict", dependencies=[Depends(limit_remote_predictions)])
async def predict_food(file: UploadFile = File(...)):

    # Validate image content type when supplied.
    if (
        file.content_type
        and not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image.",
        )

    # Read at most one byte past the limit, so an oversized upload is
    # never held in memory whole.
    contents = await file.read(MAX_UPLOAD_BYTES + 1)

    # Reject empty files.
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # Limit upload size to 10 MB.
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    # Decode the image and apply the evaluation preprocessing.
    try:
        with Image.open(BytesIO(contents)) as image:
            tensor = preprocess_image(image)
            thumbnail_jpeg = make_thumbnail(image)

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="The file is not a valid image.",
        )

    tensor = tensor.unsqueeze(0).to(DEVICE)

    # Run model inference.
    with torch.inference_mode():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(3, len(CLASS_NAMES)),
        )

    # Format the top predictions.
    top_predictions = [
        {
            "food": CLASS_NAMES[index.item()],
            "confidence": round(
                prob.item() * 100,
                2,
            ),
        }
        for prob, index in zip(
            top_probs,
            top_indices,
        )
    ]

    # Persist the prediction. A storage failure must not hide the
    # prediction, and must not be reported as a saved record.
    prediction_id = None
    storage_error = STORAGE_ERROR

    if storage_error is None:
        image_sha256 = hashlib.sha256(contents).hexdigest()
        try:
            history_store.save_thumbnail(image_sha256, thumbnail_jpeg)
        except OSError as error:
            print(f"WARNING: thumbnail not stored: {error}")

        try:
            record = history_store.add_prediction(
                filename=file.filename,
                image_sha256=image_sha256,
                predicted_food=top_predictions[0]["food"],
                confidence=top_predictions[0]["confidence"],
                top_predictions=top_predictions,
                device=str(DEVICE),
                model_name="EfficientNet-B0",
            )
            prediction_id = record["id"]
        except (OSError, sqlite3.Error) as error:
            print(f"WARNING: prediction not saved: {error}")
            storage_error = (
                "The prediction could not be saved to history. "
                "The backend console shows the cause."
            )
            try:
                history_store.remove_thumbnail_if_unused(image_sha256)
            except (OSError, sqlite3.Error):
                pass

    # Return the actual model prediction.
    return {
        "filename": file.filename,
        "predicted_food": top_predictions[0]["food"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
        "device": str(DEVICE),
        "note": (
            "Confidence is not the same as prediction accuracy."
        ),
        "low_confidence": (
            top_predictions[0]["confidence"] < LOW_CONFIDENCE_THRESHOLD
        ),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "prediction_id": prediction_id,
        "saved": prediction_id is not None,
        "storage_error": storage_error,
    }


# =========================================================
# 9. PREDICTION HISTORY ENDPOINTS
# =========================================================

@app.get("/api/history")
def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: str = Query("", max_length=100),
):
    require_storage()
    items, total = history_store.list_predictions(
        limit=limit, offset=offset, query=q,
    )
    return {
        "items": [to_api_record(item) for item in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.delete("/api/history", dependencies=[Depends(require_admin)])
def clear_history(confirm: str = Query("", max_length=10)):
    # There are no user accounts: this deletes every record this backend
    # has stored. ?confirm=all guards against an accidental request; it is
    # not access control.
    require_storage()
    if confirm != "all":
        raise HTTPException(
            status_code=400,
            detail=(
                "Deleting all prediction history needs ?confirm=all. "
                "Nothing was deleted."
            ),
        )
    deleted_records, deleted_thumbnails = history_store.clear_predictions()
    return {
        "deleted_records": deleted_records,
        "deleted_thumbnails": deleted_thumbnails,
    }


@app.patch("/api/history/{prediction_id}/feedback", dependencies=[Depends(require_admin)])
def set_feedback(prediction_id: PredictionId, feedback: FeedbackRequest):
    require_storage()
    if feedback.actual_food not in CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f'"{feedback.actual_food}" is not one of the '
                f"{len(CLASS_NAMES)} food classes."
            ),
        )
    record = history_store.set_feedback(prediction_id, feedback.actual_food)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return to_api_record(record)


@app.delete("/api/history/{prediction_id}/feedback", dependencies=[Depends(require_admin)])
def clear_feedback(prediction_id: PredictionId):
    require_storage()
    record = history_store.clear_feedback(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return to_api_record(record)


@app.delete("/api/history/{prediction_id}", dependencies=[Depends(require_admin)])
def delete_history_record(prediction_id: PredictionId):
    require_storage()
    if not history_store.delete_prediction(prediction_id):
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return {"deleted": prediction_id}


@app.get("/api/history/{prediction_id}/thumbnail")
def get_thumbnail(prediction_id: PredictionId):
    require_storage()
    record = history_store.get_prediction(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    path = history_store.thumbnail_path(record["image_sha256"])
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="No stored thumbnail for this prediction.",
        )
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


# =========================================================
# 10. LIVE EVALUATION FROM USER-VERIFIED LABELS
# =========================================================

@app.get("/api/evaluation/live")
def get_live_evaluation():
    # Counts only predictions with a user-verified actual label.
    # Kept separate from the offline test-set report at /api/evaluation.
    require_storage()
    return history_store.live_metrics()


# =========================================================
# 11. NUTRITION REFERENCE VALUES
# =========================================================

@app.get("/api/nutrition")
def get_nutrition_coverage():
    # Which food classes have USDA reference values, and how closely
    # they match. Counts come from the reference file itself.
    require_nutrition()
    return nutrition_reference.coverage()


@app.get("/api/nutrition/{food_class}")
def get_nutrition(
    food_class: str,
    grams: float = Query(
        ...,
        gt=0,
        le=MAX_GRAMS,
        allow_inf_nan=False,
        description="Serving weight in grams entered by the user (more than 0, at most 2000).",
    ),
):
    # Reference values scaled to a user-entered weight; never a measurement
    # of the photographed food, and never a portion estimated from an image.
    require_nutrition()
    if food_class not in nutrition_reference.by_class:
        raise HTTPException(
            status_code=404,
            detail=f'"{food_class}" is not one of the {len(CLASS_NAMES)} food classes.',
        )
    try:
        return nutrition_reference.estimate(food_class, grams)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))