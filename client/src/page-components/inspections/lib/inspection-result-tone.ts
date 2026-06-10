import { InspectionSegmentStatus } from "@/types/contracts";

export const getInspectionResultTone = (status: InspectionSegmentStatus) => {
  switch (status) {
    case "ok":
      return {
        badge: "success" as const,
        stroke: "#2f7d4f",
        fill: "rgba(47, 125, 79, 0.10)",
        label: "На месте",
      };
    case "missing":
      return {
        badge: "danger" as const,
        stroke: "#a23a3a",
        fill: "rgba(162, 58, 58, 0.10)",
        label: "Отсутствует",
      };
    case "extra":
      return {
        badge: "warning" as const,
        stroke: "#9a6700",
        fill: "rgba(154, 103, 0, 0.10)",
        label: "Лишнее",
      };
    case "unmatched":
      return {
        badge: "warning" as const,
        stroke: "#9a6700",
        fill: "rgba(154, 103, 0, 0.10)",
        label: "Не сопоставлено",
      };
    default:
      return {
        badge: "info" as const,
        stroke: "#57606a",
        fill: "rgba(87, 96, 106, 0.10)",
        label: status,
      };
  }
};
