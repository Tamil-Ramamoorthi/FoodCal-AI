import { useEffect, useState } from "react";
import { getNutritionCoverage } from "../services/api";
import { formatFood } from "../utils/evaluation";
import { DATA_QUALITY, MATCH_TYPES, filterFoods } from "../utils/nutrition";
import Metric from "./Metric";
import NutritionEstimator from "./NutritionEstimator";

// Nutrition lookup for any of the model's food classes, plus the coverage
// of the USDA reference data. Independent of image predictions.
export default function NutritionPage({ onAnalyze }) {
  const [coverage, setCoverage] = useState({ status: "loading", data: null, error: "" });
  const [query, setQuery] = useState("");
  const [food, setFood] = useState("");

  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    getNutritionCoverage()
      .then((data) => {
        if (!active) return;
        setCoverage({ status: "ready", data, error: "" });
        setFood((current) => current || data.foods.find((item) => item.available)?.food_class || data.foods[0]?.food_class || "");
      })
      .catch((err) => { if (active) setCoverage({ status: "error", data: null, error: err.message || "Nutrition data could not be loaded." }); });
    return () => { active = false; };
  }, [attempt]);

  function retry() {
    setCoverage({ status: "loading", data: null, error: "" });
    setAttempt((count) => count + 1);
  }

  const data = coverage.data;
  const foods = data ? data.foods.map((item) => item.food_class) : [];
  const byClass = Object.fromEntries((data?.foods || []).map((item) => [item.food_class, item]));
  const matching = filterFoods(foods, query);
  // Keep the selected food selectable even when the search hides it.
  const visible = matching.includes(food) || !food ? matching : [food, ...matching];
  const options = visible.map((name) => ({
    value: name,
    label: `${formatFood(name)}${byClass[name]?.available ? "" : " (no data)"}`,
  }));

  return (
    <>
      <section className="page-heading">
        <span className="eyebrow">NUTRITION WORKSPACE</span>
        <h1>Nutrition Lookup</h1>
        <p>Reference calories, protein, carbohydrates and fat for the {data?.total_classes ?? 34} foods the model recognizes, scaled to a serving size you enter.</p>
      </section>

      {coverage.status === "loading" && <section className="panel"><p className="nutrition-status"><span className="spinner" /> Loading nutrition data...</p></section>}
      {coverage.status === "error" && (
        <section className="panel">
          <div className="nutrition-error">
            <div className="error-banner" role="alert">Could not load nutrition data. {coverage.error}</div>
            <button className="button button-secondary" onClick={retry}>Retry</button>
          </div>
        </section>
      )}

      {data && (
        <>
          <section className="metric-grid">
            <Metric icon="♧" label="Food classes" value={data.total_classes} note="Every class the model can predict" />
            <Metric icon="✓" label="Direct matches" value={data.direct_matches} note="Reference record for the same food" />
            <Metric icon="≈" label="Approximate matches" value={data.approximate_matches} note="Related food or component, labelled as approximate" />
            <Metric icon="–" label="Unavailable" value={data.unavailable_classes} note="No reliable reference; no values shown" />
          </section>

          <div className="report-grid">
            <section className="panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">LOOK UP A FOOD</span>
                  <h2>Estimate nutrition</h2>
                  <p>Search for a food, choose it and enter your serving weight.</p>
                </div>
              </div>
              <label className="field-label" htmlFor="nutrition-search">Search foods</label>
              <input
                id="nutrition-search"
                className="text-input"
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="e.g. samosa, pizza, idli"
              />
              {query.trim() && <p className="nutrition-help">{matching.length} of {foods.length} foods match.</p>}
              {food && (
                <NutritionEstimator food={food} onFoodChange={setFood} options={options} title={`Nutrition for ${formatFood(food)}`} />
              )}
              <button className="button button-secondary nutrition-analyze" onClick={onAnalyze}>▣ Analyze a food image instead</button>
            </section>

            <section className="panel caveat-panel">
              <h2 className="panel-subtitle">About these values</h2>
              <ul className="caveat-list">
                {data.sources.map((source) => (
                  <li key={source.key}>{source.name}: {source.dataset}. {source.license}</li>
                ))}
                {data.research && (
                  <li>Which record represents each food, its match type, quality rating and limitations come from the FoodCal AI Nutrition Research Report ({data.research.generated}, in docs/research/). The research read its records through a public FoodData Central mirror; every value shown here was checked against the USDA files downloaded on {data.retrieved}.</li>
                )}
                <li>No network request is made for each prediction.</li>
                <li>Values are typical for a reference food, not a measurement of your meal. Recipes, oil, fillings and portion sizes vary.</li>
                <li><strong>Approximate</strong> means no reference exists for the exact dish, so a related food or one component of the dish is used; the reason is shown with each one.</li>
                <li><strong>Low quality</strong> references, such as a single branded product, and values <strong>flagged for review</strong> are shown with a warning.</li>
                <li><strong>Unavailable</strong> foods have no reliable reference, so no numbers are shown rather than a guess.</li>
                <li>Serving size is always what you enter. FoodCal AI does not estimate portions from photos.</li>
              </ul>
            </section>
          </div>

          <section className="panel history-page nutrition-coverage">
            <div className="section-title-row">
              <div>
                <h2>Coverage by food</h2>
                <p>{data.available_classes} of {data.total_classes} foods have reference values.</p>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th scope="col">Food</th><th scope="col">Match</th><th scope="col">Quality</th><th scope="col">Reference food</th><th scope="col">Limitations</th></tr></thead>
                <tbody>
                  {data.foods.map((item) => {
                    const match = MATCH_TYPES[item.match_type];
                    const quality = DATA_QUALITY[item.data_quality];
                    return (
                      <tr key={item.food_class}>
                        <td><button className="text-button food-name" onClick={() => setFood(item.food_class)}>{formatFood(item.food_class)}</button></td>
                        <td><span className={`status-pill ${match.style}`}><span className="status-dot" />{match.label}</span></td>
                        <td>{quality ? <span className={`status-pill ${quality.style}`}><span className="status-dot" />{quality.label}</span> : "—"}</td>
                        <td>{item.food_description ? <>{item.food_description}<span className="table-filename">{item.source_name} · {item.record_label} {item.record_id}</span></> : "—"}</td>
                        <td className="nutrition-notes-cell">{item.limitations || item.notes}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </>
  );
}
