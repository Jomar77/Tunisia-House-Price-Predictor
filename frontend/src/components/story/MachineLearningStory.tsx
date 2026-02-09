export function MachineLearningStory() {
  return (
    <div className="story-content">
      <h2>Machine Learning Pipeline</h2>
      
      <section>
        <h3>1. Data Collection</h3>
        <p>
          The journey started with web scraping Tunisian real estate listings using Python's 
          BeautifulSoup and requests libraries. The scraper was designed to extract key features:
        </p>
        <ul>
          <li><strong>Area (m²)</strong> - Property size</li>
          <li><strong>Rooms</strong> - Number of bedrooms</li>
          <li><strong>Bathrooms</strong> - Number of bathrooms</li>
          <li><strong>Age</strong> - Property age in years</li>
          <li><strong>Location</strong> - Geographic region</li>
        </ul>
      </section>

      <section>
        <h3>2. Data Cleaning & Preprocessing</h3>
        <p>
          Raw data required extensive cleaning to handle outliers and missing values. 
          Domain-specific heuristics were applied:
        </p>
        <div className="code-block">
          <code>
            # Per-location price sanity checks<br/>
            # Room-to-area ratio validation<br/>
            # Age bounds enforcement
          </code>
        </div>
        <p>
          Feature engineering included one-hot encoding for categorical location data while 
          preserving the model's ability to generalize to unseen locations through an "other" category.
        </p>
      </section>

      <section>
        <h3>3. Model Training</h3>
        <p>
          After comparing Linear Regression, Lasso, and Decision Trees, <strong>Linear Regression</strong> was 
          selected for its interpretability and performance on this dataset.
        </p>
        <p>
          The model was trained in a Jupyter notebook with cross-validation to ensure robustness.
        </p>
      </section>

      <section>
        <h3>4. Model Export</h3>
        <p>
          Instead of using pickle (security risks), we export model weights using <strong>Safetensors</strong>:
        </p>
        <div className="code-block">
          <code>
            safetensors.numpy.save_file(<br/>
            &nbsp;&nbsp;{`{`}"weights": coefficients, "bias": intercept{`}`},<br/>
            &nbsp;&nbsp;"tunisia_home_prices_model.safetensors"<br/>
            )
          </code>
        </div>
        <p>
          Alongside <code>columns.json</code> for feature ordering, ensuring deterministic inference.
        </p>
      </section>
    </div>
  );
}
