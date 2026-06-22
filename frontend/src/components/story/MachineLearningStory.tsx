export function MachineLearningStory() {
  return (
    <div className="story-content">
      <h2>Machine Learning Pipeline</h2>

      <section>
        <h3>1. Data Collection</h3>
        <p>
          This clone is being retargeted to New Zealand property data. The training pipeline is
          expected to ingest a New Zealand housing dataset and extract key features such as:
        </p>
        <ul>
          <li><strong>Area (m²)</strong> - Property size</li>
          <li><strong>Rooms</strong> - Number of bedrooms</li>
          <li><strong>Bathrooms</strong> - Number of bathrooms</li>
          <li><strong>Age</strong> - Property age in years</li>
          <li><strong>Location</strong> - New Zealand region or suburb</li>
        </ul>
      </section>

      <section>
        <h3>2. Data Cleaning & Preprocessing</h3>
        <p>
          Raw data should be cleaned with deterministic rules to reduce noisy labels and invalid
          listings. The notebook should apply location-aware sanity checks tailored to the NZ market:
        </p>
        <div className="code-block">
          <code>
            price_per_m2 = price / area<br/>
            if count(location) &lt;= 10: location = "other"<br/>
            keep if price_per_m2 fits location-specific bounds<br/>
            drop if area / rooms is implausibly low<br/>
            drop if bathrooms &gt;= rooms + 2
          </code>
        </div>
        <p>
          Age should be normalized from range strings using average semantics, i.e.
          <code> age = (a + b) / 2 </code> for inputs like <code>&quot;5-10&quot;</code>, and numeric casts for single values.
        </p>
        <p>
          A second location-specific consistency rule can remove listings where a higher-room property
          has lower <code>price_per_m2</code> than the local baseline, provided the sample count is large enough.
        </p>
      </section>

      <section>
        <h3>3. Model Training</h3>
        <p>
          The NZ clone can start with a simple baseline model and iterate from there. Linear Regression
          remains a good default if you want an interpretable model and a stable export contract.
        </p>
        <p>The optimization target is mean squared error:</p>
        <div className="code-block">
          <code>
            minimize (1 / n) * Σ(yᵢ - ŷᵢ)^2
          </code>
        </div>
        <p>The prediction equation used at inference time is:</p>
        <div className="code-block">
          <code>
            ŷ = x · w + b
          </code>
        </div>
        <p>
          Train/test split and ShuffleSplit cross-validation are used to validate generalization before export.
        </p>
      </section>

      <section>
        <h3>4. Feature Vector Contract</h3>
        <p>
          Inference strictly follows <code>columns.json</code> ordering. For each request:
        </p>
        <div className="code-block">
          <code>
            x = zeros(len(data_columns))<br/>
            x[idx("area")] = area<br/>
            x[idx("rooms")] = rooms<br/>
            x[idx("bathrooms")] = bathrooms<br/>
            x[idx("age")] = age<br/>
            x[idx(location)] = 1 (or idx("other") if unknown)
          </code>
        </div>
        <p>
          This guarantees deterministic mapping from API payloads to model input dimensions.
        </p>
      </section>

      <section>
        <h3>5. Notebook vs Production Runtime</h3>
        <p>
          The notebook helper function raises an error for unknown locations, while production backend
          inference is more robust: it falls back to the <code>other</code> one-hot column when available.
        </p>
        <p>
          This difference is intentional for API stability: exploratory notebook code is strict, runtime API
          code is fault-tolerant for unseen user input.
        </p>
      </section>

      <section>
        <h3>6. Secure Model Export</h3>
        <p>
          Instead of pickle/joblib, model parameters are exported with <strong>Safetensors</strong> using explicit keys:
        </p>
        <div className="code-block">
          <code>
            save_file({`{`}"coef": w, "intercept": [b]{`}`}, "nz_home_prices_model.safetensors")
          </code>
        </div>
        <p>
          The artifact set includes <code>columns.json</code> (feature ordering) and
          <code> model_metadata.json </code> (training metadata), making inference reproducible and safe.
        </p>
      </section>
    </div>
  );
}
