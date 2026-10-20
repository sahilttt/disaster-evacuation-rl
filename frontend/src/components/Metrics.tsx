import type { SimulationState } from "../types/api";
import { ACTION_NAMES } from "../types/api";

interface MetricsProps {
  sim: SimulationState | null;
}

export default function Metrics({ sim }: MetricsProps) {
  if (!sim) {
    return (
      <Panel title="Live Metrics">
        <div className="flex items-center justify-center h-32">
          <p className="text-slate-500 text-sm font-medium animate-pulse">
            Waiting for simulation...
          </p>
        </div>
      </Panel>
    );
  }

  const statusLabel = sim.done
    ? sim.success
      ? "Goal Reached"
      : "Failed"
    : "Running";

  const statusColor = sim.done
    ? sim.success
      ? "text-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.5)]"
      : "text-red-400 drop-shadow-[0_0_8px_rgba(248,113,113,0.5)]"
    : "text-sky-400 drop-shadow-[0_0_8px_rgba(56,189,248,0.5)]";

  return (
    <Panel title="Live Metrics">
      <div className="space-y-4">
        {/* Status Highlight */}
        <div className="flex items-center justify-between bg-slate-900/50 p-3 rounded-xl border border-slate-700/50">
          <span className="text-slate-400 text-sm font-semibold uppercase tracking-wider">Status</span>
          <span className={`text-base font-bold ${statusColor} flex items-center gap-2`}>
            {sim.done && (sim.success ? "✅" : "❌")} {statusLabel}
          </span>
        </div>

        {/* Stats Grid */}
        <dl className="grid grid-cols-2 gap-3">
          <StatBox label="Current Step" value={sim.currentStep} />
          
          <StatBox 
            label="Last Action" 
            value={sim.actionTaken !== null ? ACTION_NAMES[sim.actionTaken] ?? sim.actionTaken : "—"} 
          />
          
          <StatBox 
            label="Step Reward" 
            value={sim.reward.toFixed(1)} 
            color={sim.reward > 0 ? "text-emerald-400" : sim.reward < 0 ? "text-red-400" : "text-slate-200"}
          />
          
          <StatBox 
            label="Total Reward" 
            value={sim.totalReward.toFixed(1)} 
            color={sim.totalReward > 0 ? "text-emerald-400 font-bold" : sim.totalReward < 0 ? "text-red-400 font-bold" : "text-slate-200 font-bold"}
          />
        </dl>

        {/* Termination Reason (if done) */}
        {sim.terminationReason && (
          <div className="mt-2 bg-slate-900/40 border border-slate-800 p-3 rounded-lg text-center">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Termination Cause</p>
            <p className="text-sm font-medium text-slate-300 capitalize">
              {sim.terminationReason.replace(/_/g, " ")}
            </p>
          </div>
        )}
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ */

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl bg-slate-800/40 border border-slate-700 p-5 shadow-xl backdrop-blur-sm">
      <div className="flex items-center gap-2 mb-4">
        <div className="h-4 w-1 bg-amber-500 rounded-full"></div>
        <h2 className="text-sm font-bold text-slate-200 uppercase tracking-widest">
          {title}
        </h2>
      </div>
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */

function StatBox({
  label,
  value,
  color = "text-slate-100"
}: {
  label: string;
  value: React.ReactNode;
  color?: string;
}) {
  return (
    <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3 flex flex-col justify-center items-center text-center">
      <dt className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">{label}</dt>
      <dd className={`font-mono text-sm sm:text-base ${color}`}>{value}</dd>
    </div>
  );
}
