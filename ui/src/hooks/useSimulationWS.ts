import { useEffect, useRef, useState } from 'react';

interface WSProgress {
  progress: number;
  message: string;
  status: 'connecting' | 'listening' | 'complete' | 'stopped' | 'error';
}

export function useSimulationWS(simulationId: string | null): WSProgress {
  const [state, setState] = useState<WSProgress>({
    progress: 0,
    message: '',
    status: 'connecting',
  });
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!simulationId) return;

    const stored = localStorage.getItem('regentwin-api-url');
    const apiBase = stored || `${window.location.protocol}//${window.location.host}`;
    const wsUrl = apiBase.replace(/^http/, 'ws') + `/api/v1/simulate/${simulationId}/ws`;

    const connect = () => {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setState((s) => ({ ...s, status: 'listening' }));
      };

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.event === 'progress') {
            setState({
              progress: msg.data.percent ?? 0,
              message: msg.data.message ?? '',
              status: 'listening',
            });
          } else if (msg.event === 'complete') {
            setState((s) => ({ ...s, progress: 100, status: 'complete' }));
          } else if (msg.event === 'stopped') {
            setState((s) => ({ ...s, status: 'stopped' }));
          }
        } catch {
          // ignore malformed messages
        }
      };

      ws.onerror = () => {
        setState((s) => ({ ...s, status: 'error' }));
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [simulationId]);

  return state;
}
