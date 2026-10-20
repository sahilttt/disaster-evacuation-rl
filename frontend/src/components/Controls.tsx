interface ControlsProps {
  onReset: () => void;
  onStep: () => void;
  onRun: () => void;
  onGenerateNew: () => void;
  mode: "static" | "dynamic";
  onModeChange: (mode: "static" | "dynamic") => void;
  /** Which action is currently loading, or null. */
  loading: string | null;
  /** True when the episode is finished (disable Step). */
  done: boolean;
}

export default function Controls({
  onReset,
  onStep,
  onRun,
  onGenerateNew,
  mode,
  onModeChange,
  loading,
  done,
}: ControlsProps) {
  return (
    <div className="flex flex-col gap-4">
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3 flex flex-col gap-3">
        <div className="flex items-center justify-between px-1">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Environment Mode</span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${mode === 'dynamic' ? 'bg-amber-900/40 text-amber-400' : 'bg-slate-800 text-slate-300'}`}>
            {mode === 'dynamic' ? 'Dynamic Active' : 'Static Active'}
          </span>
        </div>
        <div className="flex bg-slate-950 rounded-lg p-1 border border-slate-800/80">
          <button 
            onClick={() => onModeChange("static")}
            className={`flex-1 text-sm py-1.5 rounded-md font-medium transition-colors ${mode === "static" ? "bg-slate-800 text-white shadow-sm" : "text-slate-400 hover:text-slate-200"}`}
          >
            Static
          </button>
          <button 
            onClick={() => onModeChange("dynamic")}
            className={`flex-1 text-sm py-1.5 rounded-md font-medium transition-colors ${mode === "dynamic" ? "bg-amber-900/60 text-amber-100 shadow-sm" : "text-slate-400 hover:text-slate-200"}`}
          >
            Dynamic
          </button>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row lg:flex-col gap-3">
        <ActionButton
          label="Generate New Scenario"
          icon="✨"
          onClick={onGenerateNew}
          busy={loading === "reset"}
          disabled={loading !== null}
          color="from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 shadow-indigo-900/50"
        />
        <ActionButton
          label="Reset Environment"
          icon="🔄"
          onClick={onReset}
          busy={loading === "reset"}
          disabled={loading !== null}
          color="from-sky-600 to-blue-600 hover:from-sky-500 hover:to-blue-500 shadow-blue-900/50"
        />
        <ActionButton
          label="Single Step"
          icon="⏩"
          onClick={onStep}
          busy={loading === "step"}
          disabled={loading !== null || done}
          color="from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 shadow-teal-900/50"
        />
        <ActionButton
          label="Run Evacuation"
          icon="🚀"
          onClick={onRun}
          busy={loading === "run"}
          disabled={loading !== null || done}
          color="from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 shadow-orange-900/50"
        />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

interface ActionButtonProps {
  label: string;
  icon: string;
  onClick: () => void;
  busy: boolean;
  disabled: boolean;
  color: string;
}

function ActionButton({
  label,
  icon,
  onClick,
  busy,
  disabled,
  color,
}: ActionButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`relative w-full flex items-center justify-center gap-2.5 px-5 py-3 rounded-xl text-sm font-semibold tracking-wide
        text-white transition-all duration-200 active:scale-[0.98]
        disabled:opacity-50 disabled:active:scale-100 disabled:cursor-not-allowed
        bg-gradient-to-r shadow-lg ${disabled ? "from-slate-700 to-slate-800 shadow-none border border-slate-700" : color}`}
    >
      {busy ? <Spinner /> : <span className="text-lg drop-shadow-sm">{icon}</span>}
      {label}
    </button>
  );
}

/* ------------------------------------------------------------------ */

function Spinner() {
  return (
    <svg
      className="animate-spin -ml-1 mr-1 h-5 w-5 text-white/80"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  );
}
