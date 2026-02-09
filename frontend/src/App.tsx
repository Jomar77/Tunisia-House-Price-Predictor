import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SplitLayout } from './components/layout/SplitLayout';
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
        <SplitLayout />
      </div>
    </QueryClientProvider>
  );
}

export default App;
