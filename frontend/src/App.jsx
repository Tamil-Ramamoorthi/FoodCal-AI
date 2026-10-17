import { Fragment, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import {
  API_BASE_URL,
  checkApi,
  clearFeedback,
  clearHistory,
  deleteHistoryRecord,
  getAdminToken,
  getFoodClasses,
  getHistory,
  getLiveEvaluation,
  isConnectionError,
  predictFood,
  saveFeedback,
  setAdminToken,
  thumbnailSrc,
} from "./services/api";
import Metric from "./components/Metric";
import ModelEvaluation from "./components/ModelEvaluation";
import NutritionEstimator from "./components/NutritionEstimator";
import NutritionPage from "./components/NutritionPage";
import { createVerification, formatFood, isLowConfidence } from "./utils/evaluation";
import { predictionOptions } from "./utils/nutrition";
import "./App.css";

const HISTORY_PAGE_SIZE = 10;
const RECENT_COUNT = 4;
const SEARCH_DELAY_MS = 300;
// Background health checks: frequent while offline so the status recovers
// once the backend has finished loading the model, rare while online.
const HEALTH_RETRY_MS = 5000;
const HEALTH_POLL_MS = 30000;

const STATUS_LABELS = { unverified: "Unverified", correct: "Correct", incorrect: "Incorrect" };
const STATUS_STYLES = { unverified: "muted", correct: "success", incorrect: "danger" };

const navigation = [
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "analyze", label: "Analyze Food", icon: "▣" },
  { id: "history", label: "Prediction History", icon: "◷" },
  { id: "evaluation", label: "Model Evaluation", icon: "▥" },
  { id: "nutrition", label: "Nutrition", icon: "♧" },
  { id: "settings", label: "Settings", icon: "☷" },
];

function App() {
  const [page, setPage] = useState("dashboard");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  // The saved history record behind the current result (null when the
  // prediction was not saved or has since been deleted).
  const [currentRecord, setCurrentRecord] = useState(null);
  // Food class used for the nutrition estimate; starts at the top match
  // and can be switched to another of the top predictions.
  const [nutritionFood, setNutritionFood] = useState("");
  const [changingFeedback, setChangingFeedback] = useState(false);
  // Saved history is loaded from the backend; it is the source of truth.
  const [history, setHistory] = useState({ status: "loading", items: [], total: 0, error: "" });
  const [historyOffset, setHistoryOffset] = useState(0);
  const [recent, setRecent] = useState({ status: "loading", items: [], error: "" });
  const [live, setLive] = useState({ status: "loading", data: null, error: "" });
  const [apiStatus, setApiStatus] = useState("checking");
  const [apiError, setApiError] = useState("");
  const [classCount, setClassCount] = useState(34);
  // Percent; reported by the backend (GET / and POST /api/predict).
  const [lowConfidenceThreshold, setLowConfidenceThreshold] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [search, setSearch] = useState("");
  const [actualLabel, setActualLabel] = useState("");
  const [feedbackError, setFeedbackError] = useState("");
  const [classes, setClasses] = useState([]);
  const [editingId, setEditingId] = useState(null);
  // What the open confirmation dialog would delete: null, or
  // { kind: "record", record } or { kind: "all" }.
  const [pendingDelete, setPendingDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [historyNotice, setHistoryNotice] = useState("");
  const [busyRecordId, setBusyRecordId] = useState(null);
  const [historyActionError, setHistoryActionError] = useState("");
  const [adminTokenInput, setAdminTokenInput] = useState(() => getAdminToken());
  const [adminTokenSet, setAdminTokenSet] = useState(() => Boolean(getAdminToken()));
  const [adminTokenNotice, setAdminTokenNotice] = useState("");
  const inputRef = useRef(null);
  // Guards against duplicate concurrent predictions and lets responses
  // for a previously selected image be discarded.
  const requestInFlight = useRef(false);
  const selectionId = useRef(0);
  // Discards history responses that arrive after a newer request.
  const historyRequest = useRef(0);
  const searchTimer = useRef(null);
  const healthCheckInFlight = useRef(false);
  // Blocks a second delete request from a repeated click.
  const deleteInFlight = useRef(false);
  // Set while the backend is unreachable, so saved data is reloaded once
  // it answers again.
  const lostConnection = useRef(false);

  // A background check leaves the current status shown until it resolves.
  async function refreshApiStatus(isActive = () => true, { background = false } = {}) {
    if (!background) setApiStatus("checking");
    try {
      const data = await checkApi();
      if (!isActive()) return;
      setApiStatus(data.device || "connected");
      setApiError("");
      if (Number.isFinite(data.low_confidence_threshold)) setLowConfidenceThreshold(data.low_confidence_threshold);
    } catch (err) {
      if (!isActive()) return;
      setApiStatus("offline");
      setApiError(err.message || "Health check failed.");
    }
  }

  async function loadHistory(offset = historyOffset, query = search) {
    const requestId = ++historyRequest.current;
    setHistory((previous) => ({ ...previous, status: "loading", error: "" }));
    try {
      const data = await getHistory({ limit: HISTORY_PAGE_SIZE, offset, query });
      if (requestId !== historyRequest.current) return;
      // A deletion can empty the last page; show the new last page.
      if (!data.items.length && offset > 0 && data.total > 0) {
        loadHistory(Math.floor((data.total - 1) / HISTORY_PAGE_SIZE) * HISTORY_PAGE_SIZE, query);
        return;
      }
      setHistoryOffset(offset);
      setHistory({ status: "ready", items: data.items, total: data.total, error: "" });
    } catch (err) {
      if (requestId !== historyRequest.current) return;
      setHistory((previous) => ({ ...previous, status: "error", error: err.message }));
    }
  }

  async function loadRecent() {
    setRecent((previous) => ({ ...previous, status: "loading", error: "" }));
    try {
      const data = await getHistory({ limit: RECENT_COUNT, offset: 0 });
      setRecent({ status: "ready", items: data.items, error: "" });
    } catch (err) {
      setRecent((previous) => ({ ...previous, status: "error", error: err.message }));
    }
  }

  async function loadLive() {
    setLive((previous) => ({ ...previous, status: "loading", error: "" }));
    try {
      const data = await getLiveEvaluation();
      setLive({ status: "ready", data, error: "" });
    } catch (err) {
      setLive((previous) => ({ ...previous, status: "error", error: err.message }));
    }
  }

  function refreshSaved() {
    return Promise.all([loadHistory(), loadRecent(), loadLive()]);
  }

  async function loadClasses(isActive = () => true) {
    try {
      const data = await getFoodClasses();
      if (isActive()) {
        setClasses(data.classes || []);
        setClassCount(data.count || data.classes?.length || 34);
      }
    } catch {
      // API status and upload remain independently usable.
    }
  }

  useEffect(() => {
    let active = true;

    async function initialize() {
      await refreshApiStatus(() => active);
      if (!active) return;
      refreshSaved();
      loadClasses(() => active);
    }

    initialize();
    return () => {
      active = false;
      clearTimeout(searchTimer.current);
    };
    // Runs once on mount; refreshSaved reads the initial page and search.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-checks the backend so the status follows it starting or stopping
  // after the page loaded. Skipped while the tab is hidden.
  useEffect(() => {
    if (apiStatus === "checking") return undefined;
    let active = true;
    const timer = setInterval(async () => {
      if (document.hidden || healthCheckInFlight.current) return;
      healthCheckInFlight.current = true;
      try {
        await refreshApiStatus(() => active, { background: true });
      } finally {
        healthCheckInFlight.current = false;
      }
    }, apiStatus === "offline" ? HEALTH_RETRY_MS : HEALTH_POLL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [apiStatus]);

  // Saved history and the class list failed to load while the backend was
  // unreachable; reload them once it answers again.
  useEffect(() => {
    if (apiStatus === "offline") {
      lostConnection.current = true;
    } else if (apiStatus !== "checking" && lostConnection.current) {
      lostConnection.current = false;
      refreshSaved();
      loadClasses();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiStatus]);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function changeSearch(value) {
    setSearch(value);
    if (value.trim() && page !== "history") setPage("history");
    clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => loadHistory(0, value), SEARCH_DELAY_MS);
  }

  function selectFile(selected) {
    if (!selected) return;

    selectionId.current += 1;
    setError("");
    setResult(null);
    setCurrentRecord(null);
    setFeedbackError("");

    if (!selected.type.startsWith("image/")) {
      setError("Please choose a valid image file.");
      return;
    }

    if (selected.size > 10 * 1024 * 1024) {
      setError("The image must be 10 MB or smaller.");
      return;
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  }

  function clearImage() {
    selectionId.current += 1;
    setFile(null);
    setPreview("");
    setResult(null);
    setCurrentRecord(null);
    setError("");
    setActualLabel("");
    setFeedbackError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  async function analyzeImage() {
    if (!file) {
      setError("Choose a food image before starting analysis.");
      return;
    }
    if (requestInFlight.current) return;

    requestInFlight.current = true;
    const requestSelection = selectionId.current;
    setLoading(true);
    setError("");
    setResult(null);
    setCurrentRecord(null);
    setChangingFeedback(false);
    setFeedbackError("");
    setActualLabel("");

    try {
      const prediction = await predictFood(file);
      if (requestSelection !== selectionId.current) return;

      setResult(prediction);
      setNutritionFood(prediction.predicted_food);
      if (Number.isFinite(prediction.low_confidence_threshold)) setLowConfidenceThreshold(prediction.low_confidence_threshold);
      setCurrentRecord(prediction.saved ? {
        id: prediction.prediction_id,
        predicted_food: prediction.predicted_food,
        verification_status: "unverified",
        actual_food: null,
      } : null);
      setApiStatus(prediction.device || "connected");
      setApiError("");
      if (prediction.saved) refreshSaved();
    } catch (err) {
      if (requestSelection !== selectionId.current) return;
      setError(err.message || "Unable to analyze this image.");
      if (isConnectionError(err)) {
        setApiStatus("offline");
        setApiError(err.message);
      } else if (err.kind === "http") {
        // The backend answered, so it is reachable even though the
        // request itself was rejected.
        setApiStatus((previous) => (previous === "offline" ? "connected" : previous));
        setApiError("");
      }
    } finally {
      requestInFlight.current = false;
      setLoading(false);
    }
  }

  // Runs a feedback/delete request for one saved record, then reloads the
  // saved history so every view shows what the database now contains.
  async function updateRecord(recordId, action) {
    setBusyRecordId(recordId);
    try {
      const updated = await action();
      if (updated?.id) {
        setCurrentRecord((previous) => (previous?.id === updated.id ? updated : previous));
      } else {
        setCurrentRecord((previous) => (previous?.id === recordId ? null : previous));
      }
      await refreshSaved();
      return "";
    } catch (err) {
      if (err.status === 404) {
        setCurrentRecord((previous) => (previous?.id === recordId ? null : previous));
        refreshSaved();
      }
      return err.message || "The change could not be saved.";
    } finally {
      setBusyRecordId(null);
    }
  }

  async function submitFeedback(verdict) {
    if (!currentRecord || busyRecordId) return;

    const outcome = createVerification(currentRecord.predicted_food, verdict, actualLabel, classes);
    if (outcome.error) {
      setFeedbackError(outcome.error);
      return;
    }

    setFeedbackError("");
    const failure = await updateRecord(currentRecord.id, () => saveFeedback(currentRecord.id, outcome.verification.actualLabel));
    if (failure) {
      setFeedbackError(failure);
    } else {
      setActualLabel("");
      setChangingFeedback(false);
    }
  }

  async function removeCurrentFeedback() {
    if (!currentRecord || busyRecordId) return;
    const failure = await updateRecord(currentRecord.id, () => clearFeedback(currentRecord.id));
    setFeedbackError(failure);
  }

  async function saveHistoryFeedback(record, actualFood) {
    setHistoryActionError("");
    const failure = await updateRecord(record.id, () => saveFeedback(record.id, actualFood));
    if (failure) setHistoryActionError(failure);
    else setEditingId(null);
  }

  async function clearHistoryFeedback(record) {
    setHistoryActionError("");
    setHistoryActionError(await updateRecord(record.id, () => clearFeedback(record.id)));
  }

  function requestDelete(target) {
    if (deleteInFlight.current) return;
    setPendingDelete(target);
    setDeleteError("");
    setHistoryNotice("");
    setHistoryActionError("");
  }

  function cancelDelete() {
    if (!deleteInFlight.current) setPendingDelete(null);
  }

  // Runs the deletion confirmed in the dialog. The dialog stays open with
  // the error when the request fails, so a failure never looks like success.
  async function confirmDelete() {
    const target = pendingDelete;
    if (!target || deleteInFlight.current) return;
    deleteInFlight.current = true;
    setDeleting(true);
    setDeleteError("");
    if (target.kind === "record") setBusyRecordId(target.record.id);

    try {
      if (target.kind === "all") {
        const { deleted_records: count } = await clearHistory();
        setCurrentRecord(null);
        setEditingId(null);
        setHistoryNotice(count
          ? `Deleted all ${count} saved prediction${count === 1 ? "" : "s"}.`
          : "There was no saved prediction history to delete.");
        await Promise.all([loadHistory(0), loadRecent(), loadLive()]);
      } else {
        const { record } = target;
        try {
          await deleteHistoryRecord(record.id);
          setHistoryNotice(`Deleted the ${formatFood(record.predicted_food)} prediction (${record.original_filename}).`);
        } catch (err) {
          if (err.status !== 404) throw err;
          setHistoryNotice("That prediction had already been deleted. The history has been refreshed.");
        }
        setCurrentRecord((previous) => (previous?.id === record.id ? null : previous));
        setEditingId((previous) => (previous === record.id ? null : previous));
        await refreshSaved();
      }
      setPendingDelete(null);
    } catch (err) {
      // After a timeout or an unreadable reply the backend may still have
      // deleted the data, so the history is reloaded to show what remains.
      const uncertain = err.kind === "timeout" || err.kind === "invalid-response";
      setDeleteError(`${err.message || "The request failed."} ${uncertain
        ? "The history has been reloaded to show what remains."
        : "Nothing was deleted."}`);
      if (uncertain) refreshSaved();
    } finally {
      deleteInFlight.current = false;
      setDeleting(false);
      setBusyRecordId(null);
    }
  }

  const liveData = live.data;
  const currentVerified = currentRecord && currentRecord.verification_status !== "unverified";

  const title = navigation.find((item) => item.id === page)?.label || "Dashboard";

  function renderUploader(compact = false) {
    return (
      <div className={`upload-layout ${compact ? "compact" : ""}`}>
        <div className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">IMAGE INPUT</span>
              <h2>Analyze Your Food</h2>
              <p>Upload a food photo for AI-powered recognition.</p>
            </div>
            <span className="session-tag">EfficientNet-B0</span>
          </div>

          <div
            className={`drop-zone ${dragging ? "dragging" : ""} ${preview ? "has-preview" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              selectFile(event.dataTransfer.files?.[0]);
            }}
          >
            {preview ? (
              <div className="image-preview">
                <img src={preview} alt="Selected food" />
                <div className="preview-overlay">
                  <span className="file-pill">✓ {file?.name}</span>
                  <button className="button button-light" onClick={clearImage}>
                    Remove image
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="upload-icon">↑</div>
                <h3>Upload your food image</h3>
                <p>Drag and drop your image here or browse local files.</p>
                <span className="file-hint">JPG, PNG, WEBP · Maximum 10 MB</span>
                <button
                  className="button button-primary"
                  onClick={() => inputRef.current?.click()}
                >
                  ▧ Choose Image
                </button>
              </>
            )}
          </div>

          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={(event) => selectFile(event.target.files?.[0])}
          />

          {file && (
            <div className="file-info">
              <span className="file-check">✓</span>
              <div className="file-details">
                <strong>{file.name}</strong>
                <span>{(file.size / (1024 * 1024)).toFixed(2)} MB · Ready to analyze</span>
              </div>
              <button className="text-button" onClick={() => inputRef.current?.click()}>
                Replace
              </button>
            </div>
          )}

          {error && <div className="error-banner" role="alert">{error}</div>}

          <div className="upload-actions">
            <button
              className="button button-secondary"
              onClick={() => inputRef.current?.click()}
            >
              ▧ {preview ? "Choose another image" : "Browse files"}
            </button>
            <button
              className="button button-primary"
              disabled={!file || loading}
              onClick={analyzeImage}
            >
              {loading ? <><span className="spinner" /> Analyzing...</> : "✦ Analyze Food"}
            </button>
          </div>

          <div className="privacy-note">
            <span>✓</span> Your image is sent to your local FoodCal AI backend for prediction.
          </div>
        </div>

        <div className="panel result-panel">
          <div className="panel-heading result-heading">
            <div>
              <span className="eyebrow">AI CLASSIFICATION</span>
              <h2>Prediction Results</h2>
            </div>
            <span className={`status-pill ${result ? "success" : "muted"}`}>
              <span className="status-dot" />
              {loading ? "Inference running" : result ? "Inference complete" : "Awaiting image"}
            </span>
          </div>

          {loading ? (
            <div className="empty-result">
              <span className="large-spinner" />
              <h3>Analyzing your food...</h3>
              <p>EfficientNet-B0 is processing your image.</p>
            </div>
          ) : result ? (
            <>
              <div className={`prediction-hero ${result.low_confidence ? "prediction-hero-uncertain" : ""}`}>
                <span className="eyebrow">{result.low_confidence ? "UNCERTAIN · BEST GUESS" : "TOP MATCH"}</span>
                <div className="prediction-main">
                  <h3>{result.predicted_food.replaceAll("_", " ")}</h3>
                  <span className={`match-badge ${result.low_confidence ? "match-badge-low" : ""}`}>{Number(result.confidence).toFixed(1)}%</span>
                </div>
                <div className="model-meta">
                  <span>◉ Model: EfficientNet-B0</span>
                  <span>✓ Device: {result.device}</span>
                </div>
              </div>

              {result.low_confidence && (
                <div className="confidence-warning" role="status">
                  <strong>Low confidence: the model is unsure about this image.</strong>
                  <span>
                    {Number(result.confidence).toFixed(1)}% is below the {Number(result.low_confidence_threshold).toFixed(0)}% warning threshold.
                    Treat the top match as a guess and compare it with the alternatives below.
                  </span>
                </div>
              )}

              <div className="hypotheses-heading">
                <h3>Top predictions</h3>
                <span>Model confidence</span>
              </div>

              <div className="prediction-list">
                {(result.top_predictions || []).map((prediction, index) => (
                  <div className="prediction-item" key={prediction.food}>
                    <div className="prediction-label">
                      <span className={`rank ${index === 0 ? "rank-first" : ""}`}>
                        {index + 1}
                      </span>
                      <strong>{prediction.food.replaceAll("_", " ")}</strong>
                      <span className="prediction-percent">
                        {Number(prediction.confidence).toFixed(1)}%
                      </span>
                    </div>
                    <div className="progress-track">
                      <div
                        className={`progress-fill ${index === 0 ? "progress-first" : ""}`}
                        style={{ width: `${Math.min(100, Math.max(0, Number(prediction.confidence)))}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {nutritionFood && (
                <NutritionEstimator
                  food={nutritionFood}
                  onFoodChange={setNutritionFood}
                  options={predictionOptions(result.top_predictions)}
                  predictedFood={result.predicted_food}
                  lowConfidence={Boolean(result.low_confidence)}
                />
              )}

              {!result.saved ? (
                <div className="feedback-box">
                  <strong>Not saved to history</strong>
                  <div className="error-banner" role="alert">{result.storage_error || "The backend did not save this prediction."}</div>
                  <p className="feedback-note">This result is shown, but it is not in your history, so feedback cannot be recorded for it.</p>
                </div>
              ) : !currentRecord ? (
                <div className="feedback-box">
                  <strong>Feedback unavailable</strong>
                  <p>This prediction was deleted from history, so feedback can no longer be recorded for it.</p>
                </div>
              ) : currentVerified && !changingFeedback ? (
                <div className="feedback-box feedback-summary" aria-live="polite">
                  <strong>Feedback saved</strong>
                  <div className="feedback-result">
                    <span className={`status-pill ${STATUS_STYLES[currentRecord.verification_status]}`}>
                      <span className="status-dot" />
                      {STATUS_LABELS[currentRecord.verification_status]}
                    </span>
                    <span>
                      Actual food: <strong>{formatFood(currentRecord.actual_food)}</strong>
                      {currentRecord.verification_status === "incorrect" && <> · model predicted <span className="food-name">{formatFood(currentRecord.predicted_food)}</span></>}
                    </span>
                  </div>
                  <p>Saved in your history and included in verified evaluation.</p>
                  <p className="feedback-note">Saving feedback does not retrain the model, so it will not change how this or any other image is classified. Corrections are reviewed before they can be used to train a new model, and a new model is used only after it is measured to perform better.</p>
                  {feedbackError && <div className="error-banner" role="alert">{feedbackError}</div>}
                  <div className="feedback-links">
                    <button className="text-button" disabled={Boolean(busyRecordId)} onClick={() => { setChangingFeedback(true); setFeedbackError(""); }}>Change feedback</button>
                    <button className="text-button text-button-muted" disabled={Boolean(busyRecordId)} onClick={removeCurrentFeedback}>Remove feedback</button>
                  </div>
                </div>
              ) : (
                <div className="feedback-box">
                  <strong>{changingFeedback ? "Change your feedback" : "Is this prediction correct?"}</strong>
                  <p>Your answer is the ground truth used for verified evaluation. The model's confidence is not.</p>
                  <div className="feedback-actions">
                    <button className="button button-primary" disabled={Boolean(busyRecordId)} onClick={() => submitFeedback("correct")}>
                      ✓ Correct
                    </button>
                    <button className="button button-secondary" disabled={Boolean(busyRecordId)} onClick={() => submitFeedback("incorrect")}>
                      Report incorrect
                    </button>
                    {changingFeedback && (
                      <button className="text-button" onClick={() => { setChangingFeedback(false); setFeedbackError(""); }}>Cancel</button>
                    )}
                  </div>
                  <label className="field-label" htmlFor="actual-food">Actual food (required to report incorrect)</label>
                  <input
                    id="actual-food"
                    className="text-input"
                    list="food-classes"
                    value={actualLabel}
                    onChange={(event) => { setActualLabel(event.target.value); setFeedbackError(""); }}
                    placeholder={classes.length ? `Choose one of the ${classes.length} food classes` : "Food classes not loaded"}
                    aria-invalid={feedbackError ? "true" : undefined}
                    aria-describedby={feedbackError ? "feedback-error" : undefined}
                  />
                  <datalist id="food-classes">
                    {classes.map((food) => <option value={formatFood(food)} key={food} />)}
                  </datalist>
                  {feedbackError && <div className="error-banner" id="feedback-error" role="alert">{feedbackError}</div>}
                </div>
              )}
              <p className="disclaimer">
                Confidence is the model's estimated probability, not a guarantee of accuracy.
                The model only recognizes {classCount} foods: a dish outside that list (for example a salad) is still matched to the closest of them, sometimes with high confidence.
                {result.saved && " This prediction and your feedback are saved in your local history."}
              </p>
            </>
          ) : (
            <div className="empty-result">
              <div className="empty-icon">✦</div>
              <h3>Your prediction will appear here</h3>
              <p>Choose a food image and select Analyze Food to see the model's real results.</p>
              <div className="empty-features">
                <span>✓ Food classification</span>
                <span>✓ Confidence scores</span>
                <span>✓ Top 3 predictions</span>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderDashboard() {
    return (
      <>
        <section className="welcome-banner">
          <div className="welcome-copy">
            <span className="ready-label"><span className="status-dot" /> AI FOOD RECOGNITION</span>
            <h2>Welcome to FoodCal AI</h2>
            <p>Discover your food with AI-powered image recognition.</p>
            <p className="welcome-subtitle">
              Upload a food photo to classify it with your trained EfficientNet-B0 model.
            </p>
            <button className="button button-primary" onClick={() => setPage("analyze")}>
              ▣ Analyze Food <span>→</span>
            </button>
          </div>
          <div className="engine-card">
            <div className="engine-top">
              <span className="eyebrow">ENGINE STATUS</span>
              <span className={`status-pill ${apiStatus !== "offline" ? "success" : "danger"}`}>
                <span className="status-dot" />
                {apiStatus === "offline" ? "Offline" : apiStatus === "checking" ? "Checking" : "Connected"}
              </span>
            </div>
            <div className="engine-model">
              <span className="engine-icon">✳</span>
              <div><strong>EfficientNet-B0</strong><span>Food image classification</span></div>
            </div>
            <div className="engine-stats">
              <div><strong>—</strong><span>Inference time</span></div>
              <div><strong>{classCount}</strong><span>Food classes</span></div>
            </div>
          </div>
        </section>

        <section className="metric-grid">
          <Metric icon="▧" label="Saved Predictions" value={liveData ? liveData.total_predictions : "—"} note={live.status === "error" ? "History unavailable" : "In your history"} />
          <Metric icon="✓" label="Correct Predictions" value={liveData?.verified ? liveData.correct : "—"} note={liveData?.verified ? `${liveData.verified} verified` : "No verified labels yet"} />
          <Metric icon="◎" label="Verified Accuracy" value={liveData?.accuracy == null ? "—" : `${(liveData.accuracy * 100).toFixed(1)}%`} note="Based on your feedback" />
          <Metric icon="✦" label="Most Predicted Food" value={liveData?.most_predicted ? formatFood(liveData.most_predicted.food) : "—"} note={liveData?.most_predicted ? `${liveData.most_predicted.count} prediction${liveData.most_predicted.count === 1 ? "" : "s"}` : "No predictions yet"} />
        </section>

        {renderUploader(true)}

        <section className="panel recent-panel">
          <div className="section-title-row">
            <div><h2>Recent Analyses</h2><p>Your latest saved food image predictions.</p></div>
            <button className="text-button" onClick={() => setPage("history")}>View history →</button>
          </div>
          {recent.status === "error" ? (
            <LoadError message={`Could not load recent analyses. ${recent.error}`} onRetry={loadRecent} />
          ) : recent.items.length ? <HistoryTable items={recent.items} lowConfidenceThreshold={lowConfidenceThreshold} /> : recent.status === "loading" ? (
            <LoadingRow label="Loading recent analyses..." />
          ) : (
            <div className="table-empty">
              <div className="empty-icon small">◷</div>
              <strong>No analyses yet</strong>
              <span>Your analyzed food images will appear here.</span>
            </div>
          )}
        </section>
      </>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => setPage("dashboard")}>
          <span className="brand-symbol">C</span>
          <span className="brand-name">FoodCal AI</span>
          <span className="version-badge">V1.0</span>
        </button>

        <div className="sidebar-section-label">PLATFORM</div>
        <nav className="side-nav">
          {navigation.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? "active" : ""}`}
              onClick={() => { setPage(item.id); setError(""); setHistoryNotice(""); }}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="api-status-line">
            <span className={`status-dot ${apiStatus === "offline" ? "dot-red" : ""}`} />
            <span>FastAPI Engine</span>
            <strong>{apiStatus === "offline" ? "Offline" : apiStatus === "checking" ? "Checking" : "Online"}</strong>
          </div>
          <div className="profile-row">
            <div className="profile-avatar">F</div>
            <div><strong>FoodCal User</strong><span>Local workspace</span></div>
          </div>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="breadcrumbs"><span>Home</span><span className="crumb-divider">/</span><strong>{title}</strong></div>
          <div className="topbar-actions">
            <span className={`status-pill ${apiStatus !== "offline" ? "success" : "danger"}`} title={apiError || undefined}>
              <span className="status-dot" />
              {apiStatus === "offline" ? "API Offline" : "EfficientNet-B0"}
            </span>
            <label className="search-box">
              <span>⌕</span>
              <input aria-label="Search predictions" placeholder="Search predictions..." value={search} onChange={(event) => changeSearch(event.target.value)} />
            </label>
            <button className="top-analyze" onClick={() => setPage("analyze")}>▣ Analyze Food</button>
          </div>
        </header>

        <main className="page-content">
          {page === "dashboard" && renderDashboard()}

          {page === "analyze" && (
            <>
              <section className="page-heading">
                <span className="eyebrow">VISION INFERENCE STUDIO</span>
                <h1>Analyze your food</h1>
                <p>Upload an image to receive a real prediction from EfficientNet-B0.</p>
              </section>
              {renderUploader(false)}
            </>
          )}

          {page === "history" && (
            <>
              <section className="page-heading"><span className="eyebrow">YOUR WORKSPACE</span><h1>Prediction History</h1><p>Saved predictions and your feedback, newest first.</p></section>
              <div className="storage-notice" role="note">
                <strong>Saved on this computer.</strong> Predictions and feedback are stored in a local SQLite database by the FoodCal AI backend, so they remain after refreshing the page or restarting the backend. Thumbnails are small copies kept in the backend's storage folder.
              </div>
              <section className="panel history-page">
                <div className="section-title-row">
                  <div>
                    <h2>All analyses</h2>
                    <p>
                      {history.status === "error" && !history.items.length ? "History could not be loaded"
                        : `${history.total} ${search.trim() ? "matching " : ""}record${history.total === 1 ? "" : "s"}${liveData && !search.trim() ? ` · ${liveData.verified} verified` : ""}`}
                    </p>
                  </div>
                  <div className="section-title-actions">
                    <button className="button button-secondary" onClick={() => refreshSaved()} disabled={history.status === "loading"}>↻ Refresh</button>
                    <button
                      className="button button-secondary button-secondary-danger"
                      disabled={deleting || Boolean(busyRecordId) || liveData?.total_predictions === 0 || (!liveData && !history.total)}
                      onClick={() => requestDelete({ kind: "all" })}
                    >
                      <TrashIcon /> Clear All History
                    </button>
                  </div>
                </div>
                {historyNotice && <div className="success-banner history-banner" role="status">{historyNotice}</div>}
                {historyActionError && <div className="error-banner history-banner" role="alert">{historyActionError}</div>}
                {history.status === "error" && (
                  <LoadError message={`Could not load saved history. ${history.error}`} onRetry={() => loadHistory()} />
                )}
                {history.items.length ? (
                  <>
                    <HistoryTable
                      lowConfidenceThreshold={lowConfidenceThreshold}
                      items={history.items}
                      actions={{
                        classes,
                        editingId,
                        busyRecordId,
                        locked: deleting,
                        onEdit: (record) => { setEditingId(record.id); setHistoryNotice(""); setHistoryActionError(""); },
                        onCancelEdit: () => setEditingId(null),
                        onSaveFeedback: saveHistoryFeedback,
                        onClearFeedback: clearHistoryFeedback,
                        onRequestDelete: (record) => requestDelete({ kind: "record", record }),
                      }}
                    />
                    <Pagination
                      offset={historyOffset}
                      total={history.total}
                      busy={history.status === "loading"}
                      onPage={(offset) => loadHistory(offset)}
                    />
                  </>
                ) : history.status === "loading" ? (
                  <LoadingRow label="Loading saved history..." />
                ) : history.status === "ready" && (
                  <div className="table-empty"><div className="empty-icon small">◷</div><strong>{search.trim() ? "No matching predictions" : "No prediction history yet"}</strong><span>{search.trim() ? "Try a different search term." : "Analyze a food image to create your first record."}</span></div>
                )}
              </section>
              {pendingDelete && (
                <ConfirmDialog
                  title={pendingDelete.kind === "all" ? "Clear all prediction history?" : "Delete this prediction?"}
                  confirmLabel={pendingDelete.kind === "all" ? "Clear All History" : "Delete prediction"}
                  busy={deleting}
                  error={deleteError}
                  onConfirm={confirmDelete}
                  onCancel={cancelDelete}
                >
                  {pendingDelete.kind === "all" ? (
                    <>
                      <p>
                        This permanently deletes {liveData ? <strong>all {liveData.total_predictions} saved prediction{liveData.total_predictions === 1 ? "" : "s"}</strong> : "every saved prediction"},
                        {" "}their feedback labels and their stored thumbnails from the FoodCal AI backend&apos;s local database.
                      </p>
                      <p>FoodCal AI has no user accounts, so this removes every record saved by this backend, not only predictions made in this browser. A search filter does not limit it.</p>
                      <p>Not affected: the EfficientNet-B0 model, its 34 food classes, nutrition data and the recorded evaluation reports. This cannot be undone.</p>
                    </>
                  ) : (
                    <>
                      <p>
                        <strong className="food-name">{formatFood(pendingDelete.record.predicted_food)}</strong>
                        {" · "}{pendingDelete.record.original_filename}
                        {" · "}<time dateTime={pendingDelete.record.created_at}>{new Date(pendingDelete.record.created_at).toLocaleString()}</time>
                      </p>
                      <p>The saved prediction and its feedback are permanently deleted. Its thumbnail is removed unless another saved prediction uses the same image. This cannot be undone.</p>
                    </>
                  )}
                </ConfirmDialog>
              )}
            </>
          )}

          {page === "evaluation" && (
            <ModelEvaluation live={live} onRetryLive={loadLive} onAnalyze={() => setPage("analyze")} />
          )}

          {page === "nutrition" && <NutritionPage onAnalyze={() => setPage("analyze")} />}

          {page === "settings" && (
            <>
              <section className="page-heading"><span className="eyebrow">APPLICATION</span><h1>Settings</h1><p>FoodCal AI configuration and system information.</p></section>
              <section className="panel settings-panel">
                <SettingRow label="Application" value="FoodCal AI" />
                <SettingRow label="Classification model" value="EfficientNet-B0" />
                <SettingRow label="Food classes" value={String(classCount)} />
                <SettingRow label="Backend API" value={API_BASE_URL} />
                <SettingRow label="Connection status" value={apiStatus === "offline" ? "Offline" : apiStatus === "checking" ? "Checking..." : "Connected"} />
                {apiStatus === "offline" && apiError && <SettingRow label="Connection error" value={apiError} />}
                <SettingRow label="Prediction history" value="Saved locally in SQLite by the backend" />
                <button className="button button-secondary" disabled={apiStatus === "checking"} onClick={() => refreshApiStatus()}>↻ Check API connection</button>
                <form
                  className="admin-token-form"
                  onSubmit={(event) => {
                    event.preventDefault();
                    setAdminToken(adminTokenInput);
                    setAdminTokenSet(Boolean(adminTokenInput.trim()));
                    setAdminTokenNotice(adminTokenInput.trim() ? "Admin token saved in this browser." : "Admin token removed from this browser.");
                  }}
                >
                  <label htmlFor="admin-token"><strong>Admin token</strong> <span>{adminTokenSet ? "Saved in this browser" : "Not set"}</span></label>
                  <p>Needed only when you use FoodCal AI through its public URL, to verify, delete or clear saved history. It must match FOODCAL_ADMIN_TOKEN on the backend and is kept only in this browser.</p>
                  <div className="admin-token-row">
                    <input id="admin-token" className="text-input" type="password" autoComplete="off" spellCheck="false" value={adminTokenInput} onChange={(event) => { setAdminTokenInput(event.target.value); setAdminTokenNotice(""); }} />
                    <button className="button button-secondary" type="submit">Save token</button>
                    {adminTokenSet && (
                      <button className="text-button text-button-muted" type="button" onClick={() => { setAdminToken(""); setAdminTokenInput(""); setAdminTokenSet(false); setAdminTokenNotice("Admin token removed from this browser."); }}>Remove</button>
                    )}
                  </div>
                  {adminTokenNotice && <div className="success-banner" role="status">{adminTokenNotice}</div>}
                </form>
              </section>
            </>
          )}
        </main>

        <footer className="app-footer">
          <span><span className={`status-dot ${apiStatus === "offline" ? "dot-red" : ""}`} /> {apiStatus === "offline" ? "API disconnected" : "Connected to FastAPI"}</span>
          <span>Model: EfficientNet-B0</span>
          <span>{classCount} food classes</span>
          <span>Predictions are estimates, not guarantees.</span>
        </footer>
      </div>
    </div>
  );
}

function Thumbnail({ record }) {
  const [failed, setFailed] = useState(false);
  const src = thumbnailSrc(record);

  // Never pretend an image exists: show a placeholder when none is stored
  // or the stored file can no longer be loaded.
  if (!src || failed) {
    return <div className="table-image" role="img" aria-label="No stored image" title="No stored image">▧</div>;
  }
  return (
    <div className="table-image">
      <img src={src} alt={`Uploaded image ${record.original_filename}`} loading="lazy" onError={() => setFailed(true)} />
    </div>
  );
}

function HistoryTable({ items, actions, lowConfidenceThreshold }) {
  const columns = actions ? 7 : 6;
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>FOOD IMAGE</th><th>PREDICTED FOOD</th><th>CONFIDENCE</th><th>TIME</th><th>ACTUAL FOOD</th><th>STATUS</th>{actions && <th>ACTIONS</th>}</tr></thead>
        <tbody>
          {items.map((item) => {
            const status = item.verification_status;
            const busy = actions?.busyRecordId === item.id;
            return <Fragment key={item.id}>
              <tr className={busy ? "row-busy" : undefined}>
                <td><Thumbnail record={item} /></td>
                <td><strong className="table-food">{formatFood(item.predicted_food)}</strong><span className="table-filename">{item.original_filename}</span></td>
                <td><div className="table-confidence"><div className="progress-track"><div className="progress-fill progress-first" style={{ width: `${item.confidence}%` }} /></div><strong>{Number(item.confidence).toFixed(1)}%</strong></div>{isLowConfidence(item.confidence, lowConfidenceThreshold) && <span className="low-confidence-tag">Low confidence</span>}</td>
                <td><time dateTime={item.created_at}>{new Date(item.created_at).toLocaleString()}</time></td>
                <td>{item.actual_food ? <strong className="table-food">{formatFood(item.actual_food)}</strong> : <span className="table-filename">Not verified</span>}</td>
                <td><span className={`status-pill ${STATUS_STYLES[status]}`}><span className="status-dot" />{STATUS_LABELS[status]}</span></td>
                {actions && (
                  <td>
                    <div className="row-actions">
                      <button className="text-button" disabled={busy || actions.locked} onClick={() => actions.onEdit(item)}>{status === "unverified" ? "Verify" : "Edit"}</button>
                      {status !== "unverified" && <button className="text-button text-button-muted" disabled={busy || actions.locked} onClick={() => actions.onClearFeedback(item)}>Undo</button>}
                      <button
                        className="text-button text-button-danger icon-text-button"
                        disabled={busy || actions.locked}
                        aria-label={`Delete ${formatFood(item.predicted_food)} prediction (${item.original_filename})`}
                        onClick={() => actions.onRequestDelete(item)}
                      >
                        <TrashIcon /> Delete
                      </button>
                    </div>
                  </td>
                )}
              </tr>
              {actions?.editingId === item.id && (
                <tr className="editor-row">
                  <td colSpan={columns}>
                    <FeedbackEditor
                      record={item}
                      classes={actions.classes}
                      busy={busy}
                      onSave={(actualFood) => actions.onSaveFeedback(item, actualFood)}
                      onCancel={actions.onCancelEdit}
                    />
                  </td>
                </tr>
              )}
            </Fragment>;
          })}
        </tbody>
      </table>
    </div>
  );
}

function FeedbackEditor({ record, classes, busy, onSave, onCancel }) {
  const [choice, setChoice] = useState(
    record.verification_status === "incorrect" ? record.actual_food : "",
  );
  const selectId = `actual-${record.id}`;

  return (
    <div className="history-editor">
      <p>Predicted <strong className="food-name">{formatFood(record.predicted_food)}</strong>. Was that right?</p>
      <div className="history-editor-actions">
        <button className="button button-primary" disabled={busy} onClick={() => onSave(record.predicted_food)}>✓ Prediction is correct</button>
        <span className="history-editor-or">or</span>
        <label className="visually-hidden" htmlFor={selectId}>Actual food</label>
        <select id={selectId} className="text-input select-input" value={choice} disabled={busy || !classes.length} onChange={(event) => setChoice(event.target.value)}>
          <option value="">{classes.length ? "Select the actual food..." : "Food classes not loaded"}</option>
          {classes.filter((food) => food !== record.predicted_food).map((food) => (
            <option key={food} value={food}>{formatFood(food)}</option>
          ))}
        </select>
        <button className="button button-secondary" disabled={busy || !choice} onClick={() => onSave(choice)}>Save as incorrect</button>
        <button className="text-button text-button-muted" onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}

function Pagination({ offset, total, busy, onPage }) {
  if (total <= HISTORY_PAGE_SIZE) return null;
  const first = offset + 1;
  const last = Math.min(offset + HISTORY_PAGE_SIZE, total);
  return (
    <nav className="pagination" aria-label="History pages">
      <span>Showing {first}–{last} of {total}</span>
      <div>
        <button className="button button-secondary" disabled={busy || offset === 0} onClick={() => onPage(Math.max(0, offset - HISTORY_PAGE_SIZE))}>← Newer</button>
        <button className="button button-secondary" disabled={busy || last >= total} onClick={() => onPage(offset + HISTORY_PAGE_SIZE)}>Older →</button>
      </div>
    </nav>
  );
}

function LoadingRow({ label }) {
  return <div className="table-empty" role="status"><span className="large-spinner" /><strong>{label}</strong></div>;
}

function LoadError({ message, onRetry }) {
  return (
    <div className="load-error">
      <div className="error-banner" role="alert">{message}</div>
      <button className="button button-secondary" onClick={onRetry}>↻ Try again</button>
    </div>
  );
}

// Modal confirmation for a destructive action. Escape cancels unless the
// request is already running; focus starts on Cancel.
function ConfirmDialog({ title, confirmLabel, busy, error, onConfirm, onCancel, children }) {
  const dialogRef = useRef(null);
  const titleId = useId();

  // A layout effect, so the dialog closes (and the browser restores focus)
  // before React removes it from the page.
  useLayoutEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog.open) dialog.showModal();
    return () => dialog.close();
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className="confirm-dialog"
      aria-labelledby={titleId}
      aria-busy={busy}
      onCancel={(event) => { event.preventDefault(); if (!busy) onCancel(); }}
    >
      <h2 id={titleId}>{title}</h2>
      <div className="confirm-dialog-body">{children}</div>
      {error && <div className="error-banner" role="alert">{error}</div>}
      <div className="confirm-dialog-actions">
        <button className="button button-secondary" disabled={busy} onClick={onCancel}>Cancel</button>
        <button className="button button-danger" disabled={busy} onClick={onConfirm}>
          {busy ? <><span className="spinner" /> Deleting...</> : <><TrashIcon /> {confirmLabel}</>}
        </button>
      </div>
    </dialog>
  );
}

function TrashIcon() {
  return (
    <svg className="trash-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M6 2.5h4M2.5 4.5h11M4 4.5l.7 8.6a1 1 0 0 0 1 .9h4.6a1 1 0 0 0 1-.9l.7-8.6M6.6 7v4.5M9.4 7v4.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SettingRow({ label, value }) {
  return <div className="setting-row"><span>{label}</span><strong>{value}</strong></div>;
}

export default App;
