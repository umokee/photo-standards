import { InspectionSegmentStatus } from "@/types/contracts";

export const getInspectionResultTone = (status: InspectionSegmentStatus) => {
  switch (status) {
    case "ok":
      return {
        badge: "success" as const,
        stroke: "#3a9b5d",
        fill: "rgba(58, 155, 93, 0.12)",
        label: "На месте",
      };
    case "missing":
      return {
        badge: "danger" as const,
        stroke: "#b84646",
        fill: "rgba(184, 70, 70, 0.12)",
        label: "Отсутствует",
      };
    case "extra":
      return {
        badge: "warning" as const,
        stroke: "#c87a00",
        fill: "rgba(200, 122, 0, 0.12)",
        label: "Лишнее",
      };
    case "unmatched":
      return {
        badge: "warning" as const,
        stroke: "#d19a30",
        fill: "rgba(209, 154, 48, 0.12)",
        label: "Не сопоставлено",
      };
    default:
      return {
        badge: "info" as const,
        stroke: "#5b7fff",
        fill: "rgba(91, 127, 255, 0.12)",
        label: status,
      };
  }
};
