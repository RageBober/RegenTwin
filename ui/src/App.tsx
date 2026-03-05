import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from './lib/queryClient';
import ErrorBoundary from './components/common/ErrorBoundary';
import Layout from './components/Layout';
import Home from './routes/Home';
import Dashboard from './routes/Dashboard';
import Results from './routes/Results';
import Analysis from './routes/Analysis';
import History from './routes/History';
import Settings from './routes/Settings';

export default function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<Home />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/results/:id" element={<Results />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/analysis/:id" element={<Analysis />} />
              <Route path="/history" element={<History />} />
              <Route path="/settings" element={<Settings />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
