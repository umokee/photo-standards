import type { InspectionRealtimeStatus, InspectionTaskResult } from "@/types/contracts";

export type InspectionFocus = {
  activeMatchKey: string | null;
  setActiveMatchKey: (value: string | null) => void;
};

export type InspectionRunOverrides = {
  image?: File | null;
  cameraId?: string | null;
};

export type InspectionOutletContext = {
  currentMode: string;

  file: File | null;
  setFile: (file: File | null) => void;

  cameraId: string | null;
  setCameraId: (cameraId: string | null) => void;

  selectedClassIds: string[];

  result: InspectionTaskResult | null;
  realtimeSessionId: string | null;
  realtimeStatus: InspectionRealtimeStatus | null;
  setRealtimeSessionId: (sessionId: string | null) => void;
  taskStatus: string | null;
  taskStage: string | null;
  taskProgress: number | null;
  isLocked: boolean;

  runDisabled: boolean;
  runLabel: string;
  onRun: (overrides?: InspectionRunOverrides) => void;

  focus: InspectionFocus;
};
