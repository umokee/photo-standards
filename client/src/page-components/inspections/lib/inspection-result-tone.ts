import { InspectionSegmentStatus } from "@/types/contracts";

const INSPECTION_RESULT_TONES = {
  ok: {
    badge: "success",
    stroke: "var(--vc-inspection-ok-stroke)",
    fill: "var(--vc-inspection-ok-fill)",
    label: "На месте",
  },
  missing: {
    badge: "danger",
    stroke: "var(--vc-inspection-missing-stroke)",
    fill: "var(--vc-inspection-missing-fill)",
    label: "Отсутствует",
  },
  extra: {
    badge: "warning",
    stroke: "var(--vc-inspection-warning-stroke)",
    fill: "var(--vc-inspection-warning-fill)",
    label: "Лишнее",
  },
  unmatched: {
    badge: "warning",
    stroke: "var(--vc-inspection-warning-stroke)",
    fill: "var(--vc-inspection-warning-fill)",
    label: "Не сопоставлено",
  },
  fallback: {
    badge: "info",
    stroke: "var(--vc-inspection-muted-stroke)",
    fill: "var(--vc-inspection-muted-fill)",
  },
} as const;

export const getInspectionResultTone = (status: InspectionSegmentStatus) => {
  switch (status) {
    case "ok":
      return INSPECTION_RESULT_TONES.ok;
    case "missing":
      return INSPECTION_RESULT_TONES.missing;
    case "extra":
      return INSPECTION_RESULT_TONES.extra;
    case "unmatched":
      return INSPECTION_RESULT_TONES.unmatched;
    default:
      return {
        ...INSPECTION_RESULT_TONES.fallback,
        label: status,
      };
  }
};
