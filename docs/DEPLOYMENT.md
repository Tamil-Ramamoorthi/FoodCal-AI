# Deploying FoodCal AI on Render

Two Render services, defined in `render.yaml` at the repository root:

| Service | Type | What it runs |
|---|---|---|
| `foodcal-ai-api` | Web service (Python) | FastAPI + the EfficientNet-B0 checkpoint on CPU |
| `foodcal-ai` | Static site | The React frontend built by Vite |

The frontend calls the API directly from the browser, so the two services
need each other's URLs: the frontend's `VITE_API_BASE_URL` and the API's
`FOODCAL_CORS_ORIGINS`.

## Backend (`foodcal-ai-api`)

| Setting | Value |
|---|---|
| Root directory | repository root (the backend reads `models/`, `reports/` and `data/metadata/` from there) |
| Python | 3.12, from `.python-version` (Render's default is newer and untested here) |
| Build command | `pip install -r backend/requirements.txt` |
| Start command | `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health check | `/` |
| Plan | Standard (2 GB RAM); see [Memory](#memory-and-plan) |

`backend/requirements.txt` holds only the runtime packages, pinned to the
tested versions, with CPU-only `torch==2.11.0+cpu` and
`torchvision==0.26.0+cpu` from the PyTorch CPU index. The root
`requirements.txt` (training and evaluation tools) is unchanged and is not
used by the deployment.

**Environment variables**

| Variable | Value | Purpose |
|---|---|---|
| `FOODCAL_CORS_ORIGINS` | the static site URL, e.g. `https://foodcal-ai.onrender.com` (comma-separate several; a trailing `/` is ignored) | Browsers may call the API only from these origins. Without it, only the local dev origins are allowed. |
| `FOODCAL_DATA_DIR` | `/var/data/foodcal` | Where the SQLite history and thumbnails are stored (created on startup) |
| `OMP_NUM_THREADS` | `1` | Keeps PyTorch to the instance's CPU allowance |
| `FOODCAL_LOW_CONFIDENCE_THRESHOLD` | optional, default `50` | Unchanged existing setting |

No secrets are needed: the API has no keys, tokens or external calls.

**Files the deployment relies on** (all committed): the checkpoint
`models/efficientnet_b0/food_efficientnet_b0_best.pth` (16 MB),
`backend/app/services/nutrition_reference.json`,
`models/efficientnet_b0/evaluation_report.json`,
`data/metadata/verification_report.json` and
`reports/real_world/real_world_evaluation.json`.

## Frontend (`foodcal-ai`)

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Build command | `npm ci && npm run build` |
| Publish directory | `dist` (relative to `frontend`) |
| Node | `NODE_VERSION=24` |

**`VITE_API_BASE_URL`** must be the API's public URL, e.g.
`https://foodcal-ai-api.onrender.com`. Vite bakes it into the bundle at build
time, so **changing it needs a redeploy of the static site** (a normal
redeploy rebuilds it). A production build
**fails on purpose** if it is missing, not `https://`, or a local address, so
a deployed frontend can never call `localhost`. Local development
(`npm run dev`) still defaults to `http://127.0.0.1:8000`.

Thumbnails in the history use the same base URL
(`${VITE_API_BASE_URL}/api/history/<id>/thumbnail`); `<img>` requests need no
CORS.

## First deployment

1. Push the branch to GitHub.
2. In Render: **New → Blueprint**, pick the repository. Render reads
   `render.yaml` and asks for the two `sync: false` values. On the first
   run the URLs are not known yet: enter the default ones Render will assign,
   `https://foodcal-ai-api.onrender.com` and `https://foodcal-ai.onrender.com`,
   or any placeholder and fix them in step 4. (If a name is taken, Render adds
   a suffix to the URL.)
3. Wait for both services to deploy. Open each service to see its real URL.
4. If a URL differs from what you entered:
   - API → Environment → `FOODCAL_CORS_ORIGINS` = the static site URL; save
     (the service restarts).
   - Static site → Environment → `VITE_API_BASE_URL` = the API URL; save and
     redeploy (it must be rebuilt).
5. Check: `https://<api>/` returns `"nutrition_reference": "ok"` and
   `"history_storage": "ok"`; the site's Settings page shows the API URL; an
   uploaded photo returns a prediction and nutrition.

## Memory and plan

Measured locally with the deployment packages (CPU-only PyTorch, one worker,
Windows; Linux figures will differ somewhat):

| State | Peak memory |
|---|---|
| Idle, model loaded | ~325 MB |
| Small image (dataset size) | ~336 MB |
| 12 MP phone photo | ~466 MB |
| 27 MP photo | ~640 MB |

Uploads are decoded at full resolution before the model's resize (the
preprocessing is unchanged on purpose), so memory grows with the photo's
pixel count. Requests are handled one at a time, so concurrent uploads do not
add up.

- **Free (512 MB, 0.1 CPU): not suitable.** Too little memory for large
  photos; it spins down after 15 minutes idle (about a minute to wake, plus
  model loading); it cannot have a disk, so history is wiped on every restart
  and spin-down.
- **Starter (512 MB, 0.5 CPU): risky.** A 12 MP photo leaves little
  headroom and larger ones can exceed the limit; Render then restarts the
  service.
- **Standard (2 GB, 1 CPU): recommended** and set in `render.yaml`.

Check current prices on <https://render.com/pricing> before choosing.

## Storage and persistence

- **With the disk in `render.yaml`** (1 GB at `/var/data`, paid plans only):
  history and thumbnails survive deploys and restarts. A service with a disk
  cannot run several instances, and each deploy has a few seconds of
  downtime.
- **Without a disk** (e.g. if you remove the `disk:` block or use Free):
  Render's filesystem is ephemeral. The app still works, but **all history
  and feedback is lost on every deploy, restart or spin-down.**
- SQLite on one instance is fine for this app; scaling out would need a
  database server instead.
- Thumbnails are about 15 KB each, so 1 GB holds tens of thousands.

## Not included

- No custom domain, authentication or rate limiting. Anyone with the URL can
  upload photos and read or delete history.
- The nutrition rebuild and training tools need files that are not committed
  (`data/external/`, datasets) and are not part of the deployment.
