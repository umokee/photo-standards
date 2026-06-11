import type { Camera } from "@/types/contracts";

type CameraSidebarDotStatus = "ok" | "warning" | "error";
type CameraBadgeType = "success" | "danger" | "info" | "warning";

type CameraStatusMeta = {
  label: string;
  dotStatus: CameraSidebarDotStatus;
  badgeType: CameraBadgeType;
  isStale: boolean;
};

const CAMERA_STATUS_STALE_AFTER_MS = 10 * 60 * 1000;
const TIMEZONE_RE = /([zZ]|[+-]\d{2}:?\d{2})$/;

export const parseServerDate = (value: string | null | undefined): Date | null => {
  if (!value) return null;

  const normalized = TIMEZONE_RE.test(value) ? value : `${value}Z`;
  const date = new Date(normalized);

  if (Number.isNaN(date.getTime())) {
    return null;
  }

  return date;
};

export const formatCameraDateTime = (value: string | null) => {
  const date = parseServerDate(value);

  if (!date) return "-";

  return new Intl.DateTimeFormat("ru-RU", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
};

export const formatCameraLastCheckedAgo = (value: string | null) => {
  const checkedAt = parseServerDate(value);

  if (!checkedAt) return "Не проверялась";

  const diffMs = Date.now() - checkedAt.getTime();
  const diffMin = Math.max(0, Math.floor(diffMs / 60_000));

  if (diffMin < 1) return "Только что";
  if (diffMin < 60) return `${diffMin} мин назад`;

  const diffHours = Math.floor(diffMin / 60);
  if (diffHours < 24) return `${diffHours} ч назад`;

  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays} д назад`;
};

export const isCameraStatusStale = (camera: Camera) => {
  const checkedAt = parseServerDate(camera.last_checked_at);

  if (!checkedAt) return false;

  return Date.now() - checkedAt.getTime() > CAMERA_STATUS_STALE_AFTER_MS;
};

export const getCameraStatusMeta = (camera: Camera): CameraStatusMeta => {
  if (!camera.is_active) {
    return {
      label: "Отключена",
      dotStatus: "warning",
      badgeType: "warning",
      isStale: false,
    };
  }

  const isStale = isCameraStatusStale(camera);

  if (isStale) {
    return {
      label: "Статус устарел",
      dotStatus: "warning",
      badgeType: "warning",
      isStale: true,
    };
  }

  switch (camera.last_status) {
    case "online":
      return {
        label: "Доступна",
        dotStatus: "ok",
        badgeType: "success",
        isStale: false,
      };

    case "offline":
      return {
        label: "Недоступна",
        dotStatus: "error",
        badgeType: "danger",
        isStale: false,
      };

    default:
      return {
        label: "Не проверялась",
        dotStatus: "warning",
        badgeType: "info",
        isStale: false,
      };
  }
};

export const buildCameraDisplayUrl = (camera: Camera) => {
  if (camera.protocol === "usb") {
    return camera.device_path || "-";
  }

  const auth = camera.username ? `${camera.username}@` : "";
  const port = camera.port ? `:${camera.port}` : "";
  const rawPath = camera.stream_path ?? camera.path ?? "";
  const path = rawPath ? (rawPath.startsWith("/") ? rawPath : `/${rawPath}`) : "";

  return `${camera.protocol}://${auth}${camera.host ?? ""}${port}${path}`;
};

export const filterCamerasBySearch = (cameras: Camera[], search: string) => {
  const query = search.trim().toLowerCase();

  if (!query) {
    return cameras;
  }

  return cameras.filter((camera) => {
    const status = getCameraStatusMeta(camera);

    const haystack = [
      camera.name,
      camera.location ?? "",
      camera.description ?? "",
      camera.protocol,
      camera.host ?? "",
      camera.device_path ?? "",
      camera.stream_path ?? "",
      status.label,
    ]
      .join(" ")
      .toLowerCase();

    return haystack.includes(query);
  });
};

export const getCameraSidebarMeta = (camera: Camera) => {
  const sourceLabel =
    camera.protocol === "usb" ? camera.device_path || "USB-камера" : camera.host || "Без адреса";

  const parts = [camera.location || sourceLabel].filter(Boolean);

  return parts;
};
