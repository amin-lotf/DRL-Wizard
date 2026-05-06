export type ActionType = "Discrete" | "Continuous" | "MultiDiscrete";
export type AlgoType = "TRPO" | "A2C" | "PPO" | "SAC" | "DQN";
export type JobStatus =
  | "queued"
  | "running"
  | "stopped"
  | "stopping"
  | "failed"
  | "finished";
export type ResultType = "train" | "evaluate";

export type ConfigValue =
  | string
  | number
  | boolean
  | null
  | ConfigValue[]
  | { [key: string]: ConfigValue };

export type ConfigRecord = Record<string, ConfigValue>;

export interface EnvMetadata {
  env_id: string;
  env_name: string;
  origin: string;
  supported_action: ActionType;
}

export interface AlgoMetadata {
  algo_id: AlgoType;
  algo_name: string;
  action_type: ActionType[];
}

export interface FieldMeta {
  label: string;
  description: string;
  type: string;
  required: boolean;
  default: ConfigValue;
  ge?: number | null;
  gt?: number | null;
  le?: number | null;
  lt?: number | null;
  min_length?: number | null;
  max_length?: number | null;
  pattern?: string | null;
  enum_choices?: string[] | null;
  order?: number | null;
}

export interface WrappedConfig<T extends ConfigRecord = ConfigRecord> {
  config: T;
  meta: Record<string, FieldMeta>;
}

export interface JobResponse {
  job_id?: number;
  status: JobStatus;
  env?: EnvMetadata | null;
  algo?: AlgoMetadata | null;
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  detail?: string | null;
}

export interface TrainingRequest {
  env_id: string;
  algo_id: AlgoType;
  general_cfg: ConfigRecord;
  log_cfg: ConfigRecord;
  algo_cfg: ConfigRecord;
}

export interface SavedRunSummary {
  run_id: string;
  run_dir: string;
  env_id: string;
  algo_id: AlgoType;
  checkpoint_label: string;
  checkpoint_path: string;
}

export interface SavedRunDiscoveryResponse {
  runs: SavedRunSummary[];
  warnings: string[];
}

export interface SavedRunDetailsResponse {
  summary: SavedRunSummary;
  env_config: ConfigRecord;
  algo_config: ConfigRecord;
  log_config: ConfigRecord;
  raw_app_config: ConfigRecord;
  eval_episodes_default: number;
}

export interface SavedRunEvaluationRequest {
  env_id: string;
  episodes?: number | null;
  render: boolean;
}

export interface SavedRunEvaluationResponse {
  average_step_reward: number;
  average_episode_reward: number;
  rendered_video_base64?: string | null;
  rendered_video_mime_type?: string | null;
  render_warning?: string | null;
}

export interface HealthResponse {
  ok: boolean;
}

export interface MetricRow {
  step: number;
  [metric: string]: number;
}
