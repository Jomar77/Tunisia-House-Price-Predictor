import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PredictionForm } from './components/PredictionForm';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="app">
        <PredictionForm />
      </div>
    </QueryClientProvider>
  );
}

export default App;
