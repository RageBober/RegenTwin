import { CheckIcon } from '@heroicons/react/24/solid';
import { motion } from 'framer-motion';

interface Step {
  label: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
}

interface StepIndicatorProps {
  steps: Step[];
  currentStep: number;
  onStepClick: (index: number) => void;
}

export default function StepIndicator({ steps, currentStep, onStepClick }: StepIndicatorProps) {
  return (
    <div className="flex items-center gap-1">
      {steps.map((step, i) => {
        const isActive = i === currentStep;
        const isCompleted = i < currentStep;
        const Icon = step.icon;

        return (
          <div key={i} className="flex items-center">
            {/* Step button */}
            <button
              onClick={() => onStepClick(i)}
              data-testid={`dashboard-step-${i}`}
              className={`relative flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium
                         transition-all duration-200
                         ${isActive
                           ? 'bg-primary-500/10 text-primary-700 dark:text-primary-300'
                           : isCompleted
                             ? 'text-primary-500/70 dark:text-primary-400/60 hover:bg-surface-2'
                             : 'text-primary-900/30 dark:text-primary-100/20 hover:bg-surface-2'
                         }`}
            >
              {isCompleted ? (
                <CheckIcon className="h-4 w-4 text-emerald-500" />
              ) : (
                <Icon className="h-4 w-4" />
              )}
              <span className="hidden sm:inline">{step.label}</span>
              {isActive && (
                <motion.div
                  layoutId="step-active"
                  className="absolute inset-0 rounded-lg border border-primary-500/20 dark:border-primary-400/15"
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
            </button>

            {/* Connector line */}
            {i < steps.length - 1 && (
              <div className={`w-6 h-px mx-0.5
                ${i < currentStep
                  ? 'bg-primary-400/40'
                  : 'bg-primary-200/20 dark:bg-primary-700/20'
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
