export const queryKeys = {
  meta: {
    constants: () => ["meta", "constants"] as const,
  },

  groups: {
    all: () => ["groups"] as const,
    detail: (groupId: string) => ["groups", groupId] as const,
  },

  standards: {
    detail: (standardId: string) => ["standards", standardId] as const,
    image: (imageId: string) => ["standards", "images", imageId] as const,
  },

  segments: {
    all: () => ["segments"] as const,
  },

  training: {
    models: (groupId: string) => ["training", groupId, "models"] as const,
    model: (modelId: string) => ["training", "model", modelId] as const,
    metricsHistory: (modelId: string) => ["training", "metrics-history", modelId] as const,
    tasks: (groupId: string) => ["training", groupId, "tasks"] as const,
    task: (taskId: string) => ["training", "task", taskId] as const,
  },

  cameras: {
    all: () => ["cameras"] as const,
    detail: (cameraId: string) => ["cameras", cameraId] as const,
    usbDevices: () => ["cameras", "usb-devices"] as const,
  },

  inspections: {
    historyRoot: () => ["inspections", "history"] as const,
    history: (groupId: string | null = null) =>
      ["inspections", "history", groupId ?? "all"] as const,
    detail: (inspectionId: string) => ["inspections", "detail", inspectionId] as const,
    realtimeStatus: (sessionId: string | null) =>
      ["inspections", "realtime", "status", sessionId] as const,
  },

  settings: {
    system_stats: () => ["settings", "system_stats"] as const,
  },
} as const;
