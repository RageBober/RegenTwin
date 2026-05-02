import { useEffect, useRef, useState } from 'react';

interface InternalWSState {
  simulationId: string | null;
  progress: number;
  message: string;
  status: 'connecting' | 'listening' | 'complete' | 'cancelled' | 'failed' | 'not_found' | 'error';
}

interface WSProgress {
  progress: number;
  message: string;
  status: 'idle' | 'connecting' | 'listening' | 'complete' | 'cancelled' | 'failed' | 'not_found' | 'error';
}

const WS_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes without messages → error

export function useSimulationWS(simulationId: string | null): WSProgress {
  const [state, setState] = useState<InternalWSState>({
    simulationId: null,
    progress: 0,
    message: '',
    status: 'connecting',
  });
  const wsRef = useRef<WebSocket | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!simulationId) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      return;
    }

    const resetTimeout = () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
        setState((current) => ({ ...current, simulationId, status: 'error', message: 'Connection timed out' }));
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
      }, WS_TIMEOUT_MS);
    };

    const stored = localStorage.getItem('regentwin-api-url');
    const apiBase = stored || `${window.location.protocol}//${window.location.host}`;
    const wsUrl = apiBase.replace(/^http/, 'ws') + `/api/v1/simulate/${simulationId}/ws`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setState({ simulationId, progress: 0, message: '', status: 'listening' });
      resetTimeout();
    };

    ws.onmessage = (e) => {
      resetTimeout();
      try {
        const msg = JSON.parse(e.data);
        if (msg.event === 'progress') {
          setState((current) => ({
            simulationId,
            progress: msg.data.percent ?? current.progress,
            message: msg.data.message ?? current.message,
            status: current.simulationId === simulationId ? current.status : 'listening',
          }));
          return;
        }

        if (msg.event === 'complete') {
          setState((current) => ({ ...current, simulationId, progress: 100, status: 'complete' }));
        } else if (msg.event === 'cancelled') {
          setState((current) => ({ ...current, simulationId, status: 'cancelled' }));
        } else if (msg.event === 'failed') {
          setState((current) => ({
            ...current,
            simulationId,
            status: 'failed',
            message: msg.data.detail ?? current.message,
          }));
        } else if (msg.event === 'not_found') {
          setState((current) => ({
            ...current,
            simulationId,
            status: 'not_found',
            message: msg.data.detail ?? current.message,
          }));
        }
      } catch {
        // Ignore malformed messages.
      }
    };

    ws.onerror = () => {
      setState((current) => ({ ...current, simulationId, status: 'error' }));
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [simulationId]);

  if (!simulationId) {
    return { progress: 0, message: '', status: 'idle' };
  }

  if (state.simulationId !== simulationId) {
    return { progress: 0, message: '', status: 'connecting' };
  }

  return {
    progress: state.progress,
    message: state.message,
    status: state.status,
  };
}
