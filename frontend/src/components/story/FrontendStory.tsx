export function FrontendStory() {
  return (
    <div className="story-content">
      <h2>Frontend Architecture</h2>
      
      <section>
        <h3>1. Tech Stack Selection</h3>
        <p>
          Built with modern React and Vite for optimal developer experience and performance:
        </p>
        <ul>
          <li><strong>React 19</strong> - Latest features and concurrent rendering</li>
          <li><strong>TypeScript</strong> - Type safety throughout the application</li>
          <li><strong>Vite</strong> - Lightning-fast HMR and optimized builds</li>
          <li><strong>TanStack Query</strong> - Server state management</li>
        </ul>
      </section>

      <section>
        <h3>2. Component Architecture</h3>
        <p>
          Following React best practices with functional components and hooks:
        </p>
        <div className="code-block">
          <code>
            // Custom hooks for separation of concerns<br/>
            const {`{`} data: metadata {`}`} = useMetadata();<br/>
            const mutation = usePrediction();
          </code>
        </div>
        <p>
          The form component uses controlled inputs with real-time validation and 
          proper error handling for a smooth user experience.
        </p>
      </section>

      <section>
        <h3>3. State Management</h3>
        <p>
          Leveraging TanStack Query for async server state:
        </p>
        <ul>
          <li>Automatic caching with 5-minute stale time for metadata</li>
          <li>Optimistic updates for prediction mutations</li>
          <li>Built-in loading and error states</li>
          <li>Request deduplication and retry logic</li>
        </ul>
      </section>

      <section>
        <h3>4. Styling Approach</h3>
        <p>
          Clean CSS with CSS custom properties for theming:
        </p>
        <div className="code-block">
          <code>
            :root {`{`}<br/>
            &nbsp;&nbsp;--primary-color: #2563eb;<br/>
            &nbsp;&nbsp;--card-bg: #ffffff;<br/>
            &nbsp;&nbsp;--shadow-lg: 0 10px 25px rgba(0,0,0,0.1);<br/>
            {`}`}
          </code>
        </div>
        <p>
          Mobile-first responsive design with proper breakpoints for all device sizes.
        </p>
      </section>

      <section>
        <h3>5. User Experience</h3>
        <p>
          Focus on accessibility and usability:
        </p>
        <ul>
          <li>Semantic HTML with proper ARIA attributes</li>
          <li>Keyboard navigation support</li>
          <li>Clear loading and error states</li>
          <li>Responsive design for mobile, tablet, and desktop</li>
        </ul>
      </section>
    </div>
  );
}
