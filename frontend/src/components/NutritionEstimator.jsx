import { useEffect, useId, useState } from "react";
import { getNutrition } from "../services/api";
import { formatFood } from "../utils/evaluation";
import {
  DATA_QUALITY,
  DEFAULT_SERVING_G,
  MATCH_TYPES,
  MAX_SERVING_G,
  NUTRIENTS,
  formatGrams,
  formatNutrient,
  nutritionRequestKey,
  nutritionView,
  nutritionWarnings,
  parseServingGrams,
  qualityWarnings,
  verificationText,
} from "../utils/nutrition";

const REQUEST_DELAY_MS = 250;

// Reference nutrition for one food class, scaled to a serving weight the
// user types in. `options` (value/label pairs) adds a food selector;
// `predictedFood` and `lowConfidence` add prediction-specific warnings.
export default function NutritionEstimator({ food, onFoodChange, options, predictedFood, lowConfidence = false, title = "Nutrition estimate" }) {
  const id = useId();
  const [servingInput, setServingInput] = useState(String(DEFAULT_SERVING_G));
  const [response, setResponse] = useState(null);
  const [retry, setRetry] = useState(0);

  const parsed = parseServingGrams(servingInput);
  const requestKey = nutritionRequestKey(food, parsed);
  const grams = parsed.grams;

  useEffect(() => {
    if (!requestKey) return undefined;
    let cancelled = false;
    const timer = setTimeout(async () => {
      setResponse({ key: requestKey, status: "loading" });
      try {
        const data = await getNutrition(food, grams);
        if (!cancelled) setResponse({ key: requestKey, status: "ready", data });
      } catch (error) {
        if (!cancelled) setResponse({ key: requestKey, status: "error", error: error.message || "Nutrition could not be loaded." });
      }
    }, REQUEST_DELAY_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [requestKey, food, grams, retry]);

  const view = nutritionView({ servingError: parsed.error, requestKey, response });
  const data = view === "ready" || view === "unavailable" ? response.data : null;
  const match = data ? MATCH_TYPES[data.match_type] : null;
  const quality = data ? DATA_QUALITY[data.data_quality] : null;
  const warnings = nutritionWarnings({ lowConfidence, food, predictedFood });
  const cautions = view === "ready" ? qualityWarnings(data) : [];
  const approximate = data?.match_type === "APPROXIMATE_MATCH";

  return (
    <section className="feedback-box nutrition-box" aria-labelledby={`${id}-title`}>
      <div className="nutrition-heading">
        <strong id={`${id}-title`}>{title}</strong>
        {match && (
          <span className="nutrition-pills">
            <span className={`status-pill ${match.style}`}><span className="status-dot" />{match.label}</span>
            {quality && <span className={`status-pill ${quality.style}`}><span className="status-dot" />{quality.label}</span>}
          </span>
        )}
      </div>
      <p>Typical reference values for the selected food, scaled to the serving weight you enter. The photo is not used to estimate the portion.</p>

      {warnings.map((warning) => (
        <div className="confidence-warning nutrition-warning" role="status" key={warning}>{warning}</div>
      ))}

      <div className="nutrition-controls">
        {options && (
          <div>
            <label className="field-label" htmlFor={`${id}-food`}>Food</label>
            <select id={`${id}-food`} className="text-input" value={food} onChange={(event) => onFoodChange?.(event.target.value)}>
              {options.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}
            </select>
          </div>
        )}
        <div>
          <label className="field-label" htmlFor={`${id}-grams`}>Serving size (grams)</label>
          <input
            id={`${id}-grams`}
            className="text-input"
            type="number"
            inputMode="decimal"
            min="0"
            max={MAX_SERVING_G}
            step="any"
            value={servingInput}
            onChange={(event) => setServingInput(event.target.value)}
            aria-invalid={parsed.error ? "true" : undefined}
            aria-describedby={`${id}-grams-help`}
          />
          <span className="nutrition-help" id={`${id}-grams-help`}>
            {parsed.error || `Weigh or estimate your portion; more than 0 and up to ${MAX_SERVING_G} g.`}
          </span>
        </div>
      </div>

      <div aria-live="polite">
        {view === "invalid" && <p className="nutrition-status">Enter a valid serving size to see nutrition values.</p>}
        {view === "loading" && <p className="nutrition-status"><span className="spinner" /> Loading nutrition for {formatFood(food)}...</p>}
        {view === "error" && (
          <div className="nutrition-error">
            <div className="error-banner" role="alert">Nutrition could not be loaded. {response.error}</div>
            <button className="button button-secondary" onClick={() => setRetry((count) => count + 1)}>Retry</button>
          </div>
        )}
        {view === "unavailable" && (
          <div className="nutrition-unavailable">
            <strong>No reliable nutrition data for {formatFood(food)}</strong>
            {data.food_name && <p className="nutrition-reference">{data.food_name}{data.preparation_method && <> · {data.preparation_method}</>}</p>}
            <p>{data.notice}</p>
            {data.limitations && <p>{data.limitations}</p>}
            <p>{data.notes}</p>
            {data.verification_status && <p>{verificationText(data.verification_status)}</p>}
          </div>
        )}
        {view === "ready" && (
          <>
            {cautions.map((caution) => (
              <div className="confidence-warning nutrition-warning" role="note" key={caution}>{caution}</div>
            ))}
            {data.food_name && (
              <p className="nutrition-reference">
                <strong>{data.food_name}</strong>{data.preparation_method && <> · {data.preparation_method}</>}
              </p>
            )}
            <div className="table-wrap">
              <table className="nutrition-table">
                <thead>
                  <tr><th scope="col">Nutrient</th><th scope="col">Per 100 g</th><th scope="col">Your serving ({formatGrams(data.serving_size_g)})</th></tr>
                </thead>
                <tbody>
                  {NUTRIENTS.map((nutrient) => (
                    <tr key={nutrient.key}>
                      <th scope="row">{nutrient.label}</th>
                      <td>{formatNutrient(data.nutrition_per_100g[nutrient.key], nutrient)}</td>
                      <td><strong>{formatNutrient(data.estimated_nutrition[nutrient.key], nutrient)}</strong></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="nutrition-source">
              Source: {data.source.dataset}, {data.source.url
                ? <a href={data.source.url} target="_blank" rel="noreferrer">“{data.source.food_description}” ({data.source.record_label} {data.source.record_id})</a>
                : <>“{data.source.food_description}” ({data.source.record_label} {data.source.record_id})</>}
              . Carbohydrate: {data.source.carbohydrate_definition.charAt(0).toLowerCase() + data.source.carbohydrate_definition.slice(1)}.
            </p>
            {data.source.release && (
              <p className="nutrition-note">
                Values checked against the downloaded {data.source.dataset_detail} file ({data.source.release}).{" "}
                {verificationText(data.verification_status)}
                {data.source.mirror_url && <> <a href={data.source.mirror_url} target="_blank" rel="noreferrer">Mirror page</a>.</>}
              </p>
            )}
            {data.limitations && (
              <p className={approximate || cautions.length ? "nutrition-approx" : "nutrition-note"}>
                <strong>{approximate ? "Approximation" : "Limitations"}:</strong> {data.limitations}
              </p>
            )}
            {data.notes && <p className="nutrition-note">{data.notes}</p>}
            <p className="disclaimer">{data.notice}</p>
          </>
        )}
      </div>
    </section>
  );
}
