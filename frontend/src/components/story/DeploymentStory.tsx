export function DeploymentStory() {
  return (
    <div className="story-content">
      <h2>Deployment Pipeline</h2>
      
      <section>
        <h3>1. Containerization</h3>
        <p>
          The application runs in a single Docker container combining frontend and backend:
        </p>
        <div className="code-block">
          <code>
            # Multi-stage build<br/>
            FROM node:20 AS frontend-build<br/>
            # Build React app<br/>
            <br/>
            FROM python:3.11-slim<br/>
            # Copy built frontend<br/>
            # Install FastAPI + dependencies<br/>
            # Serve static files + API
          </code>
        </div>
        <p>
          This approach reduces complexity and cost while maintaining clean separation of concerns.
        </p>
      </section>

      <section>
        <h3>2. Cloud Platform: Google Cloud Run</h3>
        <p>
          Serverless container deployment with key benefits:
        </p>
        <ul>
          <li><strong>Auto-scaling</strong> - Scale to zero when idle, up to N instances under load</li>
          <li><strong>Pay-per-use</strong> - Only charged for actual request time</li>
          <li><strong>HTTPS</strong> - Automatic SSL certificate management</li>
          <li><strong>Cold starts</strong> - Optimized with model preloading in lifespan</li>
        </ul>
      </section>

      <section>
        <h3>3. Environment Configuration</h3>
        <p>
          Secrets and config managed via environment variables:
        </p>
        <div className="code-block">
          <code>
            # Database connection<br/>
            DATABASE_URL=postgresql://...<br/>
            <br/>
            # CORS origins<br/>
            ALLOWED_ORIGINS=https://yourdomain.com<br/>
            <br/>
            # Model artifacts path<br/>
            ARTIFACTS_PATH=/app/artifacts
          </code>
        </div>
      </section>

      <section>
        <h3>4. Database: Neon Postgres</h3>
        <p>
          Serverless Postgres for prediction logging and feedback:
        </p>
        <ul>
          <li>Connection pooling via port 6432</li>
          <li>Async SQLAlchemy for non-blocking queries</li>
          <li>Background tasks to avoid blocking inference</li>
          <li>Automatic backups and point-in-time recovery</li>
        </ul>
      </section>

      <section>
        <h3>5. CI/CD Pipeline</h3>
        <p>
          Automated deployment workflow:
        </p>
        <ol>
          <li>Push to main branch triggers GitHub Actions</li>
          <li>Run tests and linting</li>
          <li>Build Docker image</li>
          <li>Push to Google Container Registry</li>
          <li>Deploy to Cloud Run with zero downtime</li>
        </ol>
      </section>

      <section>
        <h3>6. Monitoring & Observability</h3>
        <p>
          Production monitoring setup:
        </p>
        <ul>
          <li>Cloud Run metrics (request count, latency, errors)</li>
          <li>Structured logging for prediction events</li>
          <li>Optional: Sentry for error tracking</li>
          <li>Database query performance monitoring</li>
        </ul>
      </section>
    </div>
  );
}
