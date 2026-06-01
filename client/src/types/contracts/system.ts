export type SystemStatsResponse = {
  updated_at: string;
  system: {
    hostname: string;
    uptime_sec: number;
  };
  resources: {
    cpu_percent: number;
    cpu_count_logical: number;
    memory_used_bytes: number;
    memory_total_bytes: number;
    disk_used_bytes: number;
    disk_total_bytes: number;
  };
  gpu: {
    available: boolean;
    name: string | null;
    utilization_percent: number | null;
    memory_used_mb: number | null;
    memory_total_mb: number | null;
    temperature_c: number | null;
  };
  storage: {
    used_bytes: number;
    categories: {
      standards_bytes: number;
      inspections_bytes: number;
      models_bytes: number;
      logs_bytes: number;
      other_bytes: number;
    };
  };
};
