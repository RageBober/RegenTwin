import { Suspense, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useTranslation } from 'react-i18next';
import { useSpatialScatter } from '../../hooks/useSpatialData';

const AGENT_COLORS: Record<string, string> = {
  stem: '#2ecc71',
  macro: '#e74c3c',
  fibro: '#3498db',
  neutrophil: '#f39c12',
  endothelial: '#9b59b6',
  myofibroblast: '#1abc9c',
};

function AgentSphere({ position, color }: { position: [number, number, number]; color: string }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[1.5, 16, 16]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

function AgentsScene({ agents }: { agents: { x: number; y: number; type: string }[] }) {
  return (
    <>
      {agents.map((agent, i) => (
        <AgentSphere
          key={i}
          position={[agent.x - 50, 0, agent.y - 50]}
          color={AGENT_COLORS[agent.type] || '#999'}
        />
      ))}
    </>
  );
}

function WebGLFallback() {
  const { t } = useTranslation();
  return (
    <div className="flex h-96 items-center justify-center rounded-lg bg-slate-50 dark:bg-slate-800/50">
      <p className="text-sm text-slate-500">{t('common.error')}: WebGL not supported</p>
    </div>
  );
}

export default function SpatialView3D() {
  const { t } = useTranslation();
  const [hasWebGL] = useState(() => {
    try {
      const canvas = document.createElement('canvas');
      return !!(canvas.getContext('webgl') || canvas.getContext('webgl2'));
    } catch {
      return false;
    }
  });

  const { data: scatterData, isLoading } = useSpatialScatter({});

  // Extract agent positions from Plotly scatter data
  const agents = useMemo(() => {
    if (!scatterData?.data) return [];
    const result: { x: number; y: number; type: string }[] = [];
    for (const trace of scatterData.data) {
      const x = (trace as Record<string, unknown>).x as number[] | undefined;
      const y = (trace as Record<string, unknown>).y as number[] | undefined;
      const name = ((trace as Record<string, unknown>).name as string) || 'unknown';
      if (x && y) {
        for (let i = 0; i < x.length; i++) {
          result.push({ x: x[i], y: y[i], type: name.toLowerCase() });
        }
      }
    }
    return result;
  }, [scatterData]);

  if (!hasWebGL) return <WebGLFallback />;

  if (isLoading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary-200 border-t-primary-600" />
      </div>
    );
  }

  return (
    <div className="h-[500px] rounded-lg border border-slate-200 dark:border-slate-700">
      {/* Legend */}
      <div className="flex flex-wrap gap-3 border-b border-slate-200 px-4 py-2 dark:border-slate-700">
        {Object.entries(AGENT_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-xs text-slate-600 dark:text-slate-300">{type}</span>
          </div>
        ))}
        <span className="ml-auto text-xs text-slate-400">
          {agents.length} {t('dashboard.params.populations').toLowerCase()}
        </span>
      </div>

      <Canvas camera={{ position: [80, 60, 80], fov: 50 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[50, 100, 50]} intensity={0.8} />
        <Suspense fallback={null}>
          <AgentsScene agents={agents} />
        </Suspense>
        <OrbitControls enableDamping dampingFactor={0.1} />
        <gridHelper args={[100, 10, '#ccc', '#eee']} />
      </Canvas>
    </div>
  );
}
