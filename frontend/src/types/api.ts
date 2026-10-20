/* ------------------------------------------------------------------ */
/*  TypeScript types matching the FastAPI Pydantic schemas             */
/* ------------------------------------------------------------------ */

// Cell type constants (must match config.config.py)
export const CELL_EMPTY = 0;
export const CELL_OBSTACLE = 1;
export const CELL_FIRE = 2;
export const CELL_FLOOD = 3;
export const CELL_BLOCKED_ROAD = 4;
export const CELL_SAFE_ZONE = 5;
export const CELL_AGENT = 6;

// Action constants
export const ACTION_UP = 0;
export const ACTION_DOWN = 1;
export const ACTION_LEFT = 2;
export const ACTION_RIGHT = 3;

export const ACTION_NAMES: Record<number, string> = {
  [ACTION_UP]: "UP",
  [ACTION_DOWN]: "DOWN",
  [ACTION_LEFT]: "LEFT",
  [ACTION_RIGHT]: "RIGHT",
};

/* -------- GET /api/health ---------------------------------------- */

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  environment_ready: boolean;
}

/* -------- GET /api/results --------------------------------------- */

export interface EvalMetrics {
  success_rate: number;
  avg_reward: number;
  avg_steps: number;
  avg_path_length: number;
  disaster_terminations?: number;
  disaster_encounter_rate?: number;
}

export interface AgentSummary {
  training_episodes: number;
  training_success_rate: number;
  last_100_train_sr: number;
  evaluation: EvalMetrics;
}

export interface ResultsResponse {
  generated_at?: string;
  q_learning?: AgentSummary;
  dqn_static?: AgentSummary;
  dqn_dynamic?: AgentSummary;
}

/* -------- POST /api/simulation/reset ----------------------------- */

export interface SimResetRequest {
  grid_size?: number;
  max_steps?: number;
  start_position?: number[];
  goal_position?: number[];
  obstacles?: number[][];
  fire_cells?: number[][];
  flood_cells?: number[][];
  dynamic_disasters?: boolean;
  disaster_change_interval?: number;
  seed?: number;
  mode?: "static" | "dynamic";
  generate_new?: boolean;
}

export interface SimResetResponse {
  grid: number[][];
  grid_size: number;
  agent_position: number[];
  start_position: number[];
  goal_position: number[];
  done: boolean;
  current_step: number;
  dynamic_disasters: boolean;
}

/* -------- POST /api/simulation/step ------------------------------ */

export interface SimStepRequest {
  action?: number | null;
}

export interface SimStepResponse {
  grid: number[][];
  agent_position: number[];
  reward: number;
  total_reward: number;
  done: boolean;
  current_step: number;
  action_taken: number;
  info: Record<string, unknown>;
}

/* -------- POST /api/simulation/run ------------------------------- */

export interface SimRunRequest extends SimResetRequest {}

export interface RunStepRecord {
  step: number;
  action: number;
  agent_position: number[];
  reward: number;
  done: boolean;
  info: Record<string, unknown>;
}

export interface SimRunResponse {
  success: boolean;
  total_reward: number;
  total_steps: number;
  termination_reason: string | null;
  path: number[][];
  steps: RunStepRecord[];
  final_grid: number[][];
}

/* -------- Local UI state ----------------------------------------- */

export interface SimulationState {
  grid: number[][];
  gridSize: number;
  agentPosition: number[];
  startPosition: number[];
  goalPosition: number[];
  done: boolean;
  currentStep: number;
  reward: number;
  totalReward: number;
  actionTaken: number | null;
  success: boolean | null;
  terminationReason: string | null;
  path: number[][];
}
