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
        <h3>2. Cloud Platform: Railway</h3>
        <p>
          Railway is the deployment target for this project, providing a streamlined container workflow with managed HTTPS and practical operational defaults for full-stack services. The same Docker image serves both the React static bundle and FastAPI API, and Railway handles scaling behavior while the application keeps startup latency low by loading model artifacts during app lifespan.
        </p>
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
          Neon Postgres remains the persistence layer for prediction logging and feedback, with pooled connections, async access patterns, and background tasks used so writes do not block inference responses. This keeps latency predictable in Railway deployments while preserving the project's stateless HTTP layer and durable storage boundary.
        </p>
      </section>

      <section>
        <h3>5. CI/CD Pipeline</h3>
        <p>
          The CI/CD flow runs from GitHub Actions by executing tests and lint checks, building the Docker image, publishing the artifact, and promoting the new Railway deployment with minimal service disruption. This keeps frontend and backend versions aligned in one release unit and reduces integration drift.
        </p>
      </section>

      <section>
        <h3>6. Monitoring & Observability</h3>
        <p>
          Production monitoring combines platform-level Railway metrics with structured application logging so request latency, error trends, and prediction events can be correlated quickly during incident response. Database query behavior is also tracked to catch regressions before they impact user-facing prediction performance.
        </p>
      </section>
    </div>
  );
}
