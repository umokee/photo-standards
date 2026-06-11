import type {
  InspectionAlignmentStatus,
  InspectionStatus,
} from "@/types/contracts";

export const getInspectionHistoryStatusBadgeType = (status: InspectionStatus) => {
  if (status === "passed") return "success" as const;
  if (status === "failed") return "danger" as const;
  return "info" as const;
};

export const getInspectionHistoryAlignmentLabel = (
  status: InspectionAlignmentStatus | null
) => {
  if (!status) return "-";
  if (status === "success") return "Успешно";
  if (status === "insufficient_matches") return "Недостаточно совпадений";
  if (status === "insufficient_inliers") return "Недостаточно инлаеров";
  if (status === "homography_failed") return "Ошибка гомографии";
  return status;
};

export const getInspectionHistoryAlignmentBadgeType = (
  status: InspectionAlignmentStatus | null
) => {
  if (!status) return "info" as const;
  if (status === "success") return "success" as const;
  if (status === "homography_failed") return "danger" as const;
  return "warning" as const;
};

export const formatInspectionHistoryDateTime = (value: string) => {
  return new Date(value).toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};
