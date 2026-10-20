import type { ResultsResponse, AgentSummary } from "../types/api";

interface ResultsProps {
  results: ResultsResponse | null;
}

export default function Results({ results }: ResultsProps) {
  if (!results) {
    return null;
  }

  const agents: { key: string; label: string; data?: AgentSummary; highlight?: boolean }[] = [
    { key: "q_learning", label: "Q-Learning (Baseline)", data: results.q_learning },
    { key: "dqn_static", label: "DQN (Static)", data: results.dqn_static },
    { key: "dqn_dynamic", label: "DQN (Dynamic)", data: results.dqn_dynamic, highlight: true },
  ];

  return (
    <section className="mt-8">
      <div className="flex items-center gap-3 mb-6">
        <h2 className="text-lg font-bold text-slate-100 tracking-tight">
          Experimental Results
        </h2>
        <div className="h-px flex-1 bg-slate-800"></div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {agents.map(
          ({ key, label, data, highlight }) =>
            data && <AgentCard key={key} label={label} data={data} highlight={highlight} />
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */

function AgentCard({ label, data, highlight }: { label: string; data: AgentSummary; highlight?: boolean }) {
  const ev = data.evaluation;
  return (
    <div className={`relative overflow-hidden rounded-2xl bg-slate-900 border ${highlight ? 'border-amber-500/50 shadow-lg shadow-amber-900/20' : 'border-slate-800 shadow-xl'} p-5 transition-transform duration-300 hover:-translate-y-1`}>
      {highlight && (
        <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-amber-400 to-orange-500"></div>
      )}
      
      <h3 className={`font-bold text-lg mb-4 ${highlight ? 'text-amber-400' : 'text-slate-200'}`}>
        {label}
      </h3>

      <div className="space-y-4">
        <div>
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Training</h4>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm bg-slate-950/40 rounded-lg p-3">
            <Row label="Episodes" value={data.training_episodes} />
            <Row label="Avg Success" value={pct(data.training_success_rate)} />
            <Row label="Last-100 SR" value={pct(data.last_100_train_sr)} highlight={highlight && data.last_100_train_sr > 0.9} />
          </dl>
        </div>

        <div>
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Evaluation (Greedy)</h4>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm bg-slate-950/40 rounded-lg p-3">
            <Row label="Success Rate" value={pct(ev.success_rate)} highlight={highlight && ev.success_rate > 0.9} />
            <Row label="Avg Reward" value={ev.avg_reward.toFixed(1)} />
            <Row label="Avg Steps" value={ev.avg_steps.toFixed(1)} />
            <Row label="Path Length" value={ev.avg_path_length.toFixed(1)} />
            {ev.disaster_terminations !== undefined && (
              <Row label="Disaster Deaths" value={ev.disaster_terminations} />
            )}
            {ev.disaster_encounter_rate !== undefined && (
              <Row label="Disaster Rate" value={pct(ev.disaster_encounter_rate)} />
            )}
          </dl>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function Row({
  label,
  value,
  highlight
}: {
  label: string;
  value: string | number;
  highlight?: boolean;
}) {
  return (
    <>
      <dt className="text-slate-400 truncate">{label}</dt>
      <dd className={`font-mono text-right font-medium ${highlight ? 'text-emerald-400' : 'text-slate-200'}`}>
        {value}
      </dd>
    </>
  );
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}
