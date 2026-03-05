import { useTranslation } from 'react-i18next';
import UploadFCS from '../components/Upload/UploadFCS';
import ModelSelector from '../components/Parameters/ModelSelector';
import TherapyConfig from '../components/Parameters/TherapyConfig';
import SimulationRunner from '../components/Simulation/SimulationRunner';

export default function Dashboard() {
  const { t } = useTranslation();

  return (
    <div className="flex h-full">
      {/* Left panel — scrollable */}
      <div className="w-80 flex-shrink-0 space-y-4 overflow-y-auto border-r border-slate-200 p-4 dark:border-slate-700">
        <UploadFCS />
        <ModelSelector />
        <TherapyConfig />
      </div>

      {/* Right panel */}
      <div className="flex-1 overflow-y-auto p-6">
        <h1 className="mb-6 text-2xl font-bold text-slate-900 dark:text-white">
          {t('dashboard.title')}
        </h1>
        <SimulationRunner />
      </div>
    </div>
  );
}
