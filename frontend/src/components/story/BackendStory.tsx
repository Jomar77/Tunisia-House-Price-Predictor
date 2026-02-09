export function BackendStory() {
  return (
    <div className="story-content">
      <h2>Backend Architecture</h2>
      
      <section>
        <h3>1. Framework: FastAPI</h3>
        <p>
          Built with FastAPI for modern Python async capabilities:
        </p>
        <ul>
          <li><strong>Async/await</strong> - Non-blocking request handling</li>
          <li><strong>Pydantic</strong> - Automatic request/response validation</li>
          <li><strong>OpenAPI</strong> - Auto-generated interactive docs</li>
          <li><strong>Type hints</strong> - Full IDE support and type safety</li>
        </ul>
      </section>

      <section>
        <h3>2. Hexagonal Architecture</h3>
        <p>
          Following the Ports & Adapters pattern for clean separation:
        </p>
        <div className="code-block">
          <code>
            backend/<br/>
            ├── domain/          # Core business logic<br/>
            │   ├── predictor.py<br/>
            │   └── vectorizer.py<br/>
            └── adapters/        # External interfaces<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;├── api/         # FastAPI routes<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;└── inference/   # Model loading
          </code>
        </div>
      </section>

      <section>
        <h3>3. API Endpoints</h3>
        <p>
          Three main endpoints serve the application:
        </p>
        <ul>
          <li>
            <code>POST /api/predict</code> - Accepts property features, returns price prediction
          </li>
          <li>
            <code>GET /api/metadata</code> - Returns supported locations and model info
          </li>
          <li>
            <code>POST /api/feedback</code> - Logs user feedback (future: model retraining)
          </li>
        </ul>
      </section>

      <section>
        <h3>4. Model Inference</h3>
        <p>
          Safe, efficient model loading at startup:
        </p>
        <div className="code-block">
          <code>
            # Load once at startup (FastAPI lifespan)<br/>
            model = SafetensorsModel.load(<br/>
            &nbsp;&nbsp;"artifacts/tunisia_home_prices_model.safetensors"<br/>
            )<br/>
            <br/>
            # Feature vectorization using columns.json<br/>
            vector = vectorizer.transform(input_features)
          </code>
        </div>
        <p>
          No pickle deserialization - only safe numpy array reconstruction from Safetensors.
        </p>
      </section>

      <section>
        <h3>5. Performance Optimization</h3>
        <ul>
          <li>Model loaded once at startup, not per request</li>
          <li>Async request handling for concurrent predictions</li>
          <li>Background tasks for non-critical logging</li>
          <li>Response time typically &lt;50ms</li>
        </ul>
      </section>
    </div>
  );
}
