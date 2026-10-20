import {
  CELL_OBSTACLE,
  CELL_FIRE,
  CELL_FLOOD,
  CELL_SAFE_ZONE,
} from "../types/api";

/* ------------------------------------------------------------------ */
/*  Props                                                              */
/* ------------------------------------------------------------------ */

interface GridProps {
  /** 2-D cell-type array from the API (does NOT include the agent). */
  grid: number[][];
  /** Current [row, col] of the agent. */
  agentPosition: number[];
  /** Episode start [row, col]. */
  startPosition: number[];
  /** Goal [row, col]. */
  goalPosition: number[];
  /** Optional path overlay (list of [row, col]). */
  path?: number[][];
}

/* ------------------------------------------------------------------ */
/*  Cell styling                                                       */
/* ------------------------------------------------------------------ */

interface CellStyle {
  bg: string;
  icon: string;
  glow?: string;
  emojiClass?: string;
}

function cellStyle(
  cellType: number,
  isAgent: boolean,
  isStart: boolean,
  isGoal: boolean,
  isOnPath: boolean
): CellStyle {
  // Agent takes visual priority
  if (isAgent) {
    return {
      bg: "bg-amber-400 bg-gradient-to-br from-amber-300 to-amber-500 shadow-md shadow-amber-500/50 scale-110 z-10",
      icon: "🧑",
      glow: "ring-2 ring-amber-300 ring-offset-2 ring-offset-slate-900",
      emojiClass: "text-lg drop-shadow-sm",
    };
  }

  // Goal
  if (isGoal) {
    return { 
      bg: "bg-emerald-500 bg-gradient-to-br from-emerald-400 to-emerald-600 shadow-lg shadow-emerald-500/30", 
      icon: "🏁",
      emojiClass: "text-lg animate-bounce drop-shadow-md",
    };
  }

  // Hazards / obstacles
  switch (cellType) {
    case CELL_OBSTACLE:
      return { 
        bg: "bg-slate-700 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0IiBoZWlnaHQ9IjQiPgo8cmVjdCB3aWR0aD0iNCIgaGVpZ2h0PSI0IiBmaWxsPSIjMzMzIj48L3JlY3Q+CjxyZWN0IHdpZHRoPSIxIiBoZWlnaHQ9IjEiIGZpbGw9IiM0NDQiPjwvcmVjdD4KPC9zdmc+')] shadow-inner border border-slate-600/50", 
        icon: "" 
      };
    case CELL_FIRE:
      return { 
        bg: "bg-red-950/80 border border-red-500/30", 
        icon: "🔥",
        glow: "shadow-[inset_0_0_15px_rgba(239,68,68,0.4)]",
        emojiClass: "animate-pulse drop-shadow-[0_0_8px_rgba(239,68,68,0.8)]"
      };
    case CELL_FLOOD:
      return { 
        bg: "bg-blue-950/80 border border-blue-500/30", 
        icon: "🌊",
        glow: "shadow-[inset_0_0_15px_rgba(59,130,246,0.4)]",
        emojiClass: "drop-shadow-[0_0_8px_rgba(59,130,246,0.8)]"
      };
    case CELL_SAFE_ZONE:
      return { bg: "bg-emerald-500", icon: "🏁" };
    default:
      break;
  }

  // Start marker (visible once the agent has moved away)
  if (isStart) {
    return { bg: "bg-slate-800/80", icon: "📍", emojiClass: "opacity-50 text-sm" };
  }

  // Path cells
  if (isOnPath) {
    return { 
      bg: "bg-slate-800", 
      icon: "•",
      emojiClass: "text-amber-500/50 text-xl font-bold"
    };
  }

  // Empty
  return { bg: "bg-slate-800/40 border border-slate-700/30", icon: "" };
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function Grid({
  grid,
  agentPosition,
  startPosition,
  goalPosition,
  path = [],
}: GridProps) {
  const size = grid.length;

  // Build a fast-lookup set for path membership
  const pathSet = new Set(path.map(([r, c]) => `${r},${c}`));

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-[600px] mx-auto">
      {/* Grid */}
      <div
        className="w-full aspect-square inline-grid gap-1.5 rounded-2xl border border-slate-700/50 bg-slate-900/80 p-3 shadow-2xl backdrop-blur-sm"
        style={{
          gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))`,
        }}
      >
        {grid.map((row, r) =>
          row.map((cell, c) => {
            const isAgent =
              r === agentPosition[0] && c === agentPosition[1];
            const isStart =
              r === startPosition[0] && c === startPosition[1];
            const isGoal =
              r === goalPosition[0] && c === goalPosition[1];
            const isOnPath = pathSet.has(`${r},${c}`);

            const style = cellStyle(cell, isAgent, isStart, isGoal, isOnPath);

            return (
              <div
                key={`${r}-${c}`}
                className={`relative flex items-center justify-center rounded-xl
                  transition-all duration-300 ease-in-out
                  ${style.bg} ${style.glow ?? ""}`}
                title={`(${r}, ${c})`}
              >
                <span className={`select-none ${style.emojiClass ?? ""}`}>
                  {style.icon}
                </span>
              </div>
            );
          })
        )}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center justify-center gap-4 text-xs font-medium text-slate-400 bg-slate-900/50 px-6 py-3 rounded-full border border-slate-800/80 shadow-inner">
        <LegendItem icon="🧑" label="Agent" />
        <LegendItem icon="🏁" label="Goal" />
        <LegendItem icon="📍" label="Start" />
        <LegendItem icon="🔥" label="Fire" />
        <LegendItem icon="🌊" label="Flood" />
        <LegendItem icon="▓" label="Obstacle" />
        <LegendItem icon="•" label="Path" color="text-amber-500/50" />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function LegendItem({ icon, label, color = "text-slate-300" }: { icon: string; label: string; color?: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`text-base leading-none drop-shadow-sm ${color}`}>{icon}</span>
      <span className="tracking-wide">{label}</span>
    </span>
  );
}

