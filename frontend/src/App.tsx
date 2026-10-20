import React, { useEffect, useState, useCallback, useRef } from "react";

import type {
  HealthResponse,
  ResultsResponse,
  SimulationState,
} from "./types/api";
import * as api from "./services/api";

import Header from "./components/Header";
import Grid from "./components/Grid";
import Controls from "./components/Controls";
import Metrics from "./components/Metrics";
import Results from "./components/Results";

/* ================================================================== */

export default function App() {
  /* ---- State ----------------------------------------------------- */

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [sim, setSim] = useState<SimulationState | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  /* ---- Handlers -------------------------------------------------- */

  const [mode, setMode] = useState<"static" | "dynamic">("dynamic");
  const scenarioRef = React.useRef<{
    obstacles?: number[][];
    fire_cells?: number[][];
    flood_cells?: number[][];
    start_position?: number[];
    goal_position?: number[][];
  } | null>(null);

  const handleReset = useCallback(async (generateNew: boolean = false) => {
    setLoading("reset");
    setError(null);
    try {
      let reqBody: any = { mode, generate_new: generateNew };
      if (!generateNew && scenarioRef.current) {
        reqBody = { ...reqBody, ...scenarioRef.current };
      }
      
      const res = await api.resetSimulation(reqBody);
      
      // Store the true layout to persist it across resets, avoiding defaults
      const newObstacles: number[][] = [];
      const newFire: number[][] = [];
      const newFlood: number[][] = [];
      for (let r = 0; r < res.grid_size; r++) {
        for (let c = 0; c < res.grid_size; c++) {
          if (res.grid[r][c] === 1) newObstacles.push([r, c]);
          if (res.grid[r][c] === 2) newFire.push([r, c]);
          if (res.grid[r][c] === 3) newFlood.push([r, c]);
        }
      }
      scenarioRef.current = {
        obstacles: newObstacles,
        fire_cells: newFire,
        flood_cells: newFlood,
        start_position: res.start_position,
        goal_position: res.goal_position as any,
      };

      setSim({
        grid: res.grid,
        gridSize: res.grid_size,
        agentPosition: res.agent_position,
        startPosition: res.start_position,
        goalPosition: res.goal_position,
        done: res.done,
        currentStep: res.current_step,
        reward: 0,
        totalReward: 0,
        actionTaken: null,
        success: null,
        terminationReason: null,
        path: [],
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to reset simulation");
    } finally {
      setLoading(null);
    }
  }, [mode]);

  const handleStep = useCallback(async () => {
    setLoading("step");
    setError(null);
    try {
      const res = await api.stepSimulation();
      setSim((prev) =>
        prev
          ? {
              ...prev,
              grid: res.grid,
              agentPosition: res.agent_position,
              done: res.done,
              currentStep: res.current_step,
              reward: res.reward,
              totalReward: res.total_reward,
              actionTaken: res.action_taken,
              success:
                (res.info?.reason as string) === "goal_reached"
                  ? true
                  : res.done
                    ? false
                    : null,
              terminationReason:
                (res.info?.termination as string) ?? null,
              path: [...prev.path, res.agent_position],
            }
          : null
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to step");
    } finally {
      setLoading(null);
    }
  }, []);

  const handleRun = useCallback(async () => {
    setLoading("run");
    setError(null);
    try {
      let reqBody: any = { mode };
      if (scenarioRef.current) {
        reqBody = { ...reqBody, ...scenarioRef.current };
      }
      const res = await api.runSimulation(reqBody);
      setSim((prev) =>
        prev
          ? {
              ...prev,
              grid: res.final_grid,
              agentPosition: res.path[res.path.length - 1],
              done: true,
              currentStep: res.total_steps,
              reward: 0,
              totalReward: res.total_reward,
              actionTaken: null,
              success: res.success,
              terminationReason: res.termination_reason,
              path: res.path,
            }
          : null
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to run simulation");
    } finally {
      setLoading(null);
    }
  }, [mode]);

  /* ---- Initialisation ------------------------------------------- */

  const hasInitialized = React.useRef(false);

  useEffect(() => {
    if (hasInitialized.current) return;
    hasInitialized.current = true;

    const init = async () => {
      try {
        const h = await api.getHealth();
        setHealth(h);
      } catch {
        setError("Cannot connect to backend");
      }

      try {
        const r = await api.getResults();
        setResults(r);
      } catch {
        /* results are optional */
      }

      await handleReset();
    };
    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ---- Render ---------------------------------------------------- */

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.3),rgba(255,255,255,0))]">
      <Header health={health} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8 space-y-10">
        {/* Error banner */}
        {error && (
          <div className="bg-red-950/80 border border-red-700/50 shadow-lg text-red-200 px-5 py-3 rounded-xl text-sm flex items-center gap-3 backdrop-blur-sm">
            <span className="text-xl">⚠️</span>
            <p className="font-medium">{error}</p>
          </div>
        )}

        {/* Top section: Grid + Sidebar */}
        <div className="flex flex-col lg:flex-row gap-8 items-start">
          {/* Grid area */}
          <div className="flex-1 w-full flex justify-center lg:justify-end">
            {sim ? (
              <Grid
                grid={sim.grid}
                agentPosition={sim.agentPosition}
                startPosition={sim.startPosition}
                goalPosition={sim.goalPosition}
                path={sim.path}
              />
            ) : (
              <div className="w-full max-w-[600px] aspect-square rounded-2xl border border-slate-800 bg-slate-900/50 flex flex-col items-center justify-center text-slate-500 gap-4 shadow-inner">
                <div className="animate-spin h-8 w-8 border-4 border-slate-700 border-t-slate-400 rounded-full"></div>
                <p className="font-medium animate-pulse">Initializing simulation...</p>
              </div>
            )}
          </div>

          {/* Sidebar: Controls + Metrics */}
          <div className="w-full lg:w-80 space-y-6 flex-shrink-0">
            <Controls
              onReset={() => handleReset(false)}
              onStep={handleStep}
              onRun={handleRun}
              onGenerateNew={() => handleReset(true)}
              mode={mode}
              onModeChange={setMode}
              loading={loading}
              done={sim?.done ?? true}
            />
            <Metrics sim={sim} />
          </div>
        </div>

        {/* Results section */}
        <Results results={results} />
      </main>

      {/* Footer */}
      <footer className="text-center text-xs text-slate-500 py-6 border-t border-slate-800/60 mt-auto bg-slate-950/50">
        <p>Disaster Evacuation Route Planning &middot; Reinforcement Learning Dashboard</p>
      </footer>
    </div>
  );
}
