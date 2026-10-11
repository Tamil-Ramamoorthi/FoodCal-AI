import { useEffect, useState } from "react";
import { getEvaluationReport, getRealWorldEvaluation } from "../services/api";
import { MIN_CLASS_SAMPLES, activeConfusionMatrix, formatFood, withClassRates } from "../utils/evaluation";
import Metric from "./Metric";

const percent = (value) => `${(value * 100).toFixed(1)}%`;
const rate = (value) => (value === null ? "—" : percent(value));

// Live sample sizes below this are flagged as highly uncertain.
const SMALL_SAMPLE = 30;

export default function ModelEvaluation({ live, onRetryLive, onAnalyze }) {
  const [report, setReport] = useState({ status: "loading", data: null, error: "" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    getEvaluationReport()
      .then((data) => { if (active) setReport({ status: "loaded", data, error: "" }); })
      .catch((err) => { if (active) setReport({ status: "error", data: null, error: err.message }); });
    return () => { active = false; };
  }, [attempt]);

  function retry() {
    setReport({ status: "loading", data: null, error: "" });
    setAttempt((count) => count + 1);
  }

  const [realWorld, setRealWorld] = useState({ status: "loading", data: null, error: "" });
  const [realWorldAttempt, setRealWorldAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    getRealWorldEvaluation()
      .then((data) => { if (active) setRealWorld({ status: "loaded", data, error: "" }); })
      .catch((err) => { if (active) setRealWorld({ status: "error", data: null, error: err.message }); });
    return () => { active = false; };
  }, [realWorldAttempt]);

  function retryRealWorld() {
    setRealWorld({ status: "loading", data: null, error: "" });
    setRealWorldAttempt((count) => count + 1);
  }

  return (
    <>
      <section className="page-heading">
        <span className="eyebrow">MODEL PERFORMANCE</span>
        <h1>Model Evaluation</h1>
        <p>Three separate sources: the recorded offline test-set report, a recorded evaluation on independent real-world photos, and predictions you verified in your history. They measure different things and are never combined.</p>
      </section>

      <section className="eval-section">
        <div className="eval-section-heading">
          <span className="eyebrow">SOURCE 1 · OFFLINE TEST SET</span>
          <h2>Recorded evaluation report</h2>
          <p>Computed by <code>training/evaluate_efficientnet.py</code> on the held-out test split and served unchanged by the backend.</p>
        </div>

        {report.status === "loading" && (
          <div className="panel table-empty"><span className="large-spinner" /><strong>Loading the recorded report...</strong></div>
        )}
        {report.status === "error" && (
          <div className="panel">
            <div className="error-banner" role="alert">Could not load the offline evaluation report. {report.error}</div>
            <button className="button button-secondary report-retry" onClick={retry}>↻ Try again</button>
          </div>
        )}
        {report.status === "loaded" && <OfflineReport data={report.data} />}
      </section>

      <section className="eval-section">
        <div className="eval-section-heading">
          <span className="eyebrow">SOURCE 2 · REAL-WORLD PHOTOS</span>
          <h2>Independent real-world evaluation</h2>
          <p>Computed by <code>evaluation/evaluate_real_world.py</code> on manually labeled photos in <code>data/real_world_test/</code> and served unchanged by the backend. Nothing here is estimated from model confidence.</p>
        </div>

        {realWorld.status === "loading" && (
          <div className="panel table-empty"><span className="large-spinner" /><strong>Loading the real-world report...</strong></div>
        )}
        {realWorld.status === "error" && (
          <div className="panel">
            <div className="error-banner" role="alert">Could not load the real-world evaluation report. {realWorld.error}</div>
            <button className="button button-secondary report-retry" onClick={retryRealWorld}>↻ Try again</button>
          </div>
        )}
        {realWorld.status === "loaded" && (
          realWorld.data?.status === "ok"
            ? <RealWorldReport data={realWorld.data} />
            : <RealWorldEmpty data={realWorld.data} />
        )}
      </section>

      <section className="eval-section">
        <div className="eval-section-heading">
          <span className="eyebrow">SOURCE 3 · LIVE USER FEEDBACK</span>
          <h2>Verified predictions in your history</h2>
          <p>Counted only from saved predictions where you confirmed or corrected the actual food. Model confidence is never treated as correctness.</p>
        </div>

        {live.status === "loading" && !live.data && (
          <div className="panel table-empty"><span className="large-spinner" /><strong>Loading verified feedback...</strong></div>
        )}
        {live.status === "error" && (
          <div className="panel">
            <div className="error-banner" role="alert">Could not load live feedback metrics. {live.error}</div>
            <button className="button button-secondary report-retry" onClick={onRetryLive}>↻ Try again</button>
          </div>
        )}
        {live.data && <LiveFeedback data={live.data} onAnalyze={onAnalyze} />}

        <div className="panel evaluation-note">
          <div className="empty-icon">▥</div>
          <h2>User feedback is not a test set</h2>
          <p>
            The images you choose to analyze and verify form a small, self-selected sample.
            It is useful for spotting problems, but it is not a substitute for a representative, independently labeled test dataset.
          </p>
        </div>
      </section>
    </>
  );
}

function LiveFeedback({ data, onAnalyze }) {
  const perClass = withClassRates(data.per_class);

  return (
    <>
      <div className="metric-grid">
        <Metric icon="▧" label="Verified Predictions" value={data.verified} note={`of ${data.total_predictions} saved predictions`} />
        <Metric icon="✓" label="Correct" value={data.verified ? data.correct : "—"} note="Prediction matched your label" />
        <Metric icon="×" label="Incorrect" value={data.verified ? data.incorrect : "—"} note="Prediction differed from your label" />
        <Metric icon="◎" label="Verified Accuracy" value={data.accuracy === null ? "—" : percent(data.accuracy)} note={data.verified ? `${data.correct} of ${data.verified} verified` : "No verified labels yet"} />
      </div>

      {data.verified === 0 ? (
        <div className="panel table-empty">
          <div className="empty-icon small">▥</div>
          <strong>No verified predictions yet</strong>
          <span>Analyze a food image, then mark the prediction as correct or incorrect. Live metrics appear only after that.</span>
          <button className="button button-primary report-retry" onClick={onAnalyze}>▣ Analyze Food</button>
        </div>
      ) : (
        <div className="panel recent-panel">
          <div className="section-title-row">
            <div>
              <h2>Per-class results (live)</h2>
              <p>Rates are shown once a class has at least {MIN_CLASS_SAMPLES} verified samples; below that only counts are shown.</p>
            </div>
          </div>
          {data.verified < SMALL_SAMPLE && (
            <p className="table-note">Based on only {data.verified} verified prediction{data.verified === 1 ? "" : "s"}, so this estimate is highly uncertain.</p>
          )}
          <div className="table-wrap">
            <table>
              <thead><tr><th>FOOD</th><th>VERIFIED IMAGES</th><th>CORRECT</th><th>RECALL</th><th>PREDICTED AS</th><th>PRECISION</th></tr></thead>
              <tbody>
                {perClass.map((row) => (
                  <tr key={row.food}>
                    <td><strong className="table-food">{formatFood(row.food)}</strong></td>
                    <td>{row.actual}</td>
                    <td>{row.correct}</td>
                    <td>{rate(row.recall)}</td>
                    <td>{row.predicted}</td>
                    <td>{rate(row.precision)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}

function OfflineReport({ data }) {
  const splits = data.dataset_splits;
  const perClass = [...data.per_class].sort((a, b) => a.f1 - b.f1);

  return (
    <>
      <div className="metric-grid">
        <Metric icon="◎" label="Test Accuracy" value={percent(data.accuracy)} note="Top prediction matched the test label" />
        <Metric icon="▥" label="Macro F1" value={percent(data.macro_f1)} note="Every class weighted equally" />
        <Metric icon="⚖" label="Weighted F1" value={percent(data.weighted_f1)} note="Classes weighted by test images" />
        <Metric icon="▧" label="Test Images" value={data.test_images.toLocaleString()} note={`${data.num_classes} food classes`} />
      </div>

      <div className="report-grid">
        <div className="panel settings-panel">
          <h2 className="panel-subtitle">About this report</h2>
          <div className="setting-row"><span>Model</span><strong>{data.model_name}</strong></div>
          <div className="setting-row"><span>Report file</span><strong>{data.report_file}</strong></div>
          <div className="setting-row"><span>File last modified</span><strong>{new Date(data.report_modified).toLocaleString()}</strong></div>
          {splits && (
            <div className="setting-row"><span>Dataset split (images)</span><strong>train {splits.train?.toLocaleString()} · validation {splits.validation?.toLocaleString()} · test {splits.test?.toLocaleString()}</strong></div>
          )}
          {data.checkpoint_epoch != null && (
            <div className="setting-row"><span>Checkpoint</span><strong>epoch {data.checkpoint_epoch}{data.checkpoint_val_accuracy != null ? ` · validation accuracy ${percent(data.checkpoint_val_accuracy)}` : ""}</strong></div>
          )}
          <div className="setting-row"><span>Report classes match deployed model</span><strong>{data.classes_match_model ? "Yes" : "No — report may be for a different model"}</strong></div>
        </div>

        <div className="panel">
          <h2 className="panel-subtitle">What the metrics mean</h2>
          <dl className="definition-list">
            <dt>Accuracy</dt><dd>Share of test images whose top prediction matched the true label.</dd>
            <dt>Precision</dt><dd>Of the images predicted as a food, the share that really were that food.</dd>
            <dt>Recall</dt><dd>Of the images that really were a food, the share the model found.</dd>
            <dt>F1</dt><dd>Harmonic mean of precision and recall for one class.</dd>
            <dt>Macro / weighted F1</dt><dd>Average F1 across classes, either equally or by number of test images.</dd>
          </dl>
        </div>
      </div>

      <div className="panel evaluation-note caveat-panel">
        <h2>What this report does not prove</h2>
        <ul className="caveat-list">
          <li>The test images come from the same source dataset as the training images, split at random per class. They are not photos of real meals taken with your camera.</li>
          <li>A duplicate check on 10 Oct 2026 found 96 of the 3,602 test images to be exact copies of training images, so these scores are likely slightly optimistic.</li>
          <li>Real-world accuracy remains unknown until the model is evaluated on an independently labeled set of real photos.</li>
        </ul>
      </div>

      <div className="panel recent-panel">
        <div className="section-title-row">
          <div><h2>Per-class results (offline test set)</h2><p>Sorted by F1, weakest classes first.</p></div>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>FOOD</th><th>PRECISION</th><th>RECALL</th><th>F1</th><th>TEST IMAGES</th></tr></thead>
            <tbody>
              {perClass.map((row) => (
                <tr key={row.food}>
                  <td><strong className="table-food">{formatFood(row.food)}</strong></td>
                  <td>{percent(row.precision)}</td>
                  <td>{percent(row.recall)}</td>
                  <td>
                    <div className="table-confidence">
                      <div className="progress-track"><div className="progress-fill progress-first" style={{ width: `${row.f1 * 100}%` }} /></div>
                      <strong>{percent(row.f1)}</strong>
                    </div>
                  </td>
                  <td>{row.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

const INDEPENDENCE = {
  checks_passed: { label: "Checks passed", tone: "success" },
  provenance_incomplete: { label: "Provenance incomplete", tone: "muted" },
  overlap_detected: { label: "Overlap with training data", tone: "danger" },
  overlap_not_checked: { label: "Overlap not checked", tone: "muted" },
  not_assessed: { label: "Not assessed", tone: "muted" },
};

function QualityWarnings({ quality }) {
  return (
    <div className="panel evaluation-note caveat-panel">
      <h2>Dataset quality warnings</h2>
      {quality.warnings.length ? (
        <ul className="caveat-list">{quality.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
      ) : (
        <p>No warnings were raised by the dataset checks.</p>
      )}
      {quality.duplicate_detection_limits && <p className="quality-limits">{quality.duplicate_detection_limits}</p>}
    </div>
  );
}

// Shown when no report exists yet, or the latest report found no valid images.
function RealWorldEmpty({ data }) {
  return (
    <>
      <div className="panel table-empty">
        <div className="empty-icon small">▣</div>
        <strong>{data ? "No real-world photos have been evaluated" : "No real-world evaluation report yet"}</strong>
        <span>
          {data
            ? `The latest run (${new Date(data.generated_at).toLocaleString()}) found no valid images, so no real-world metrics exist.`
            : "Real-world accuracy is unknown until the evaluation script has been run on labeled photos."}
        </span>
        <ol className="setup-steps">
          <li>Add your own photos (JPG, JPEG, PNG or WEBP) to <code>data/real_world_test/&lt;class name&gt;/</code>, one folder per food.</li>
          <li>Record where each photo came from in <code>data/real_world_test/provenance.csv</code> (columns <code>file</code>, <code>source</code>).</li>
          <li>Run <code>python -m evaluation.evaluate_real_world</code> from the project root, then reload this page.</li>
        </ol>
      </div>
      {data && data.quality.warnings.length > 0 && <QualityWarnings quality={data.quality} />}
    </>
  );
}

function RealWorldReport({ data }) {
  const { metrics, quality, dataset, checkpoint } = data;
  const classCount = data.class_mapping.length;
  const independence = INDEPENDENCE[quality.independence_status] ?? { label: quality.independence_status, tone: "muted" };
  const excluded = data.metrics_excluding_overlaps;
  const invalidCount = dataset.invalid_images.length;
  // Classes with images first (weakest F1 first), then classes without images.
  const perClass = [...metrics.per_class].sort((a, b) => (
    (a.f1 === null) - (b.f1 === null) || (a.f1 ?? 0) - (b.f1 ?? 0) || b.support - a.support
  ));
  const cm = activeConfusionMatrix(metrics.confusion_matrix);

  return (
    <>
      <div className="metric-grid">
        <Metric icon="▧" label="Evaluated Images" value={metrics.total.toLocaleString()} note={`${invalidCount} invalid file${invalidCount === 1 ? "" : "s"} excluded`} />
        <Metric icon="◎" label="Real-World Accuracy" value={percent(metrics.accuracy)} note={`${metrics.correct} of ${metrics.total} correct`} />
        <Metric icon="▥" label="Macro F1" value={percent(metrics.macro_f1)} note={`Over ${metrics.macro_f1_classes} of ${classCount} classes with images`} />
        <Metric icon="⚖" label="Weighted F1" value={percent(metrics.weighted_f1)} note="Classes weighted by images" />
      </div>

      {metrics.total < SMALL_SAMPLE && (
        <div className="panel"><p className="table-note flush">Based on only {metrics.total} image{metrics.total === 1 ? "" : "s"}, so these scores are highly uncertain.</p></div>
      )}

      <div className="report-grid real-world-grid">
        <div className="panel settings-panel">
          <h2 className="panel-subtitle">About this report</h2>
          <div className="setting-row"><span>Report file</span><strong>{data.report_file}</strong></div>
          <div className="setting-row"><span>Generated</span><strong>{new Date(data.generated_at).toLocaleString()}</strong></div>
          <div className="setting-row"><span>Dataset folder</span><strong>{dataset.path}</strong></div>
          <div className="setting-row"><span>Checkpoint</span><strong>{checkpoint.path}{checkpoint.epoch != null ? ` · epoch ${checkpoint.epoch}` : ""}</strong></div>
          <div className="setting-row"><span>Checkpoint matches deployed model</span><strong>{data.checkpoint_matches_deployed ? "Yes" : "No — re-run the evaluation"}</strong></div>
          <div className="setting-row"><span>Report classes match deployed model</span><strong>{data.classes_match_model ? "Yes" : "No"}</strong></div>
          <div className="setting-row"><span>Classes without images</span><strong>{dataset.classes_without_images.length} of {classCount}</strong></div>
          <div className="setting-row"><span>Classes below {dataset.min_images_per_class} images</span><strong>{dataset.classes_below_minimum.length}</strong></div>
        </div>

        <div className="panel settings-panel">
          <h2 className="panel-subtitle">Independence checks</h2>
          <div className="setting-row"><span>Status</span><span className={`status-pill ${independence.tone}`}><span className="status-dot" />{independence.label}</span></div>
          <p className="quality-note">{quality.independence_note}</p>
          <div className="setting-row"><span>Exact copies of training-source images</span><strong>{quality.exact_overlaps.length}</strong></div>
          <div className="setting-row"><span>Possible near-duplicates of training-source images</span><strong>{quality.near_duplicate_overlaps.length}</strong></div>
          <div className="setting-row"><span>Duplicate groups within the set</span><strong>{quality.exact_duplicates_within.length} exact · {quality.near_duplicates_within.length} possible</strong></div>
          <div className="setting-row"><span>Photos with documented source</span><strong>{quality.provenance.documented_images} of {metrics.total}</strong></div>
          {excluded && (
            <div className="setting-row"><span>Excluding {data.overlapping_images_excluded} overlapping image{data.overlapping_images_excluded === 1 ? "" : "s"}</span><strong>accuracy {percent(excluded.accuracy)} · macro F1 {rate(excluded.macro_f1)} on {excluded.total} images</strong></div>
          )}
        </div>
      </div>

      <QualityWarnings quality={quality} />

      <div className="panel recent-panel">
        <div className="section-title-row">
          <div><h2>Per-class results (real-world photos)</h2><p>Weakest classes first. &ldquo;—&rdquo; means undefined: recall and F1 need at least one image of the class, precision needs at least one prediction of it.</p></div>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>FOOD</th><th>IMAGES</th><th>PREDICTED AS</th><th>PRECISION</th><th>RECALL</th><th>F1</th></tr></thead>
            <tbody>
              {perClass.map((row) => (
                <tr key={row.food}>
                  <td><strong className="table-food">{formatFood(row.food)}</strong></td>
                  <td>{row.support}{row.support > 0 && row.support < dataset.min_images_per_class ? " (few)" : ""}</td>
                  <td>{row.predicted}</td>
                  <td>{rate(row.precision)}</td>
                  <td>{rate(row.recall)}</td>
                  <td>{rate(row.f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel recent-panel">
        <div className="section-title-row">
          <div><h2>Confusion matrix</h2><p>Rows are the actual food, columns the predicted food. Only the {cm.labels.length} classes that occur as an actual or predicted label are shown; the diagonal holds correct predictions.</p></div>
        </div>
        <div className="table-wrap">
          <table className="confusion-table">
            <thead>
              <tr><th>ACTUAL \ PREDICTED</th>{cm.labels.map((label) => <th key={label}>{formatFood(label)}</th>)}</tr>
            </thead>
            <tbody>
              {cm.matrix.map((row, i) => (
                <tr key={cm.labels[i]}>
                  <td><strong className="table-food">{formatFood(cm.labels[i])}</strong></td>
                  {row.map((count, j) => {
                    const share = cm.rowTotals[i] ? count / cm.rowTotals[i] : 0;
                    const color = i === j ? "8, 115, 47" : "220, 38, 38";
                    return (
                      <td key={cm.labels[j]} className={count ? undefined : "confusion-zero"} style={count ? { background: `rgba(${color}, ${(0.08 + share * 0.42).toFixed(2)})` } : undefined}>
                        {count || "·"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
