import type {
  Angle,
  Architecture,
  InspectionMode,
  InspectionStatus,
  Metric,
  TrainingStatus,
  UserRole,
} from "@/types/contracts";
import { ru } from "@/utils/ru";

export const roleLabel = (role: UserRole) =>
  ru.users.roles[role as keyof typeof ru.users.roles] ?? role;

export const angleLabel = (angle: Angle) =>
  ru.standards.angles[angle as keyof typeof ru.standards.angles] ?? angle;

export const inspectionModeLabel = (mode: InspectionMode) =>
  ru.inspections.modes[mode as keyof typeof ru.inspections.modes] ?? mode;

export const inspectionStatusLabel = (status: InspectionStatus) =>
  ru.inspections.statuses[status as keyof typeof ru.inspections.statuses] ?? status;

export const trainingStatusLabel = (status: TrainingStatus) =>
  ru.training.statuses[status as keyof typeof ru.training.statuses] ?? status;

export const architectureLabel = (architecture: Architecture) =>
  ru.training.architectures[architecture as keyof typeof ru.training.architectures] ??
  architecture;

export const metricLabel = (metric: Metric) => ru.training.metrics[metric];
