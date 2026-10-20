import type { HealthResponse } from "../types/api";

interface HeaderProps {
  health: HealthResponse | null;
}

export default function Header({ health }: HeaderProps) {
  const isOnline = health?.status === "ok";
  const modelOk = health?.model_loaded ?? false;
  const envOk = health?.environment_ready ?? false;

  return (
    <header className="bg-slate-900/80 backdrop-blur-md border-b border-slate-700 sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Title & Subtitle */}
        <div className="flex flex-col">
          <div className="flex items-center gap-3">
            <span className="text-2xl bg-gradient-to-br from-amber-400 to-red-500 bg-clip-text text-transparent">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-7 h-7 text-amber-500 inline-block drop-shadow-md">
                <path fillRule="evenodd" d="M14.615 1.595a.75.75 0 01.359.852L12.982 9.75h7.268a.75.75 0 01.548 1.262l-10.5 11.25a.75.75 0 01-1.272-.71l1.992-7.302H3.75a.75.75 0 01-.548-1.262l10.5-11.25a.75.75 0 01.913-.143z" clipRule="evenodd" />
              </svg>
            </span>
            <h1 className="text-xl md:text-2xl font-extrabold tracking-tight text-slate-100">
              AI Disaster Evacuation
            </h1>
          </div>
          <p className="text-sm font-medium text-slate-400 mt-1 md:ml-10">
            Route Planning Using Deep Reinforcement Learning
          </p>
        </div>

        {/* Status indicators */}
        <div className="flex items-center gap-4 text-xs font-medium bg-slate-950/50 px-4 py-2 rounded-full border border-slate-800">
          <StatusDot label="Backend" ok={isOnline} />
          <StatusDot label="Model" ok={modelOk} />
          <StatusDot label="Env" ok={envOk} />
        </div>
      </div>
    </header>
  );
}

/* ------------------------------------------------------------------ */

function StatusDot({ label, ok }: { label: string; ok: boolean }) {
  return (
    <span className="flex items-center gap-2">
      <span className="relative flex h-2.5 w-2.5">
        {ok && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-20"></span>
        )}
        <span
          className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
            ok ? "bg-emerald-500" : "bg-red-500"
          }`}
        />
      </span>
      <span className={ok ? "text-slate-300" : "text-slate-500"}>{label}</span>
    </span>
  );
}
