import Select from "@/components/ui/select/select";
import { useGetCameras } from "@/page-components/cameras/api/get-cameras";
import { useCameraStatusLive } from "@/page-components/cameras/hooks/use-camera-status-live";
import { getCameraStatusMeta } from "@/page-components/cameras/lib/camera-view";
import type { InspectionModePath } from "@/app/paths";
import { Camera, FileImage, Upload, Video } from "lucide-react";
import { useEffect, useMemo, useRef, type ChangeEvent } from "react";
import { DEVICE_CAMERA_OPTION, isDeviceCameraId } from "../../lib/device-camera";
import s from "./inspection-source-control.module.scss";

const MODES_REQUIRING_CAMERA = new Set<InspectionModePath>(["snapshot", "realtime"]);

type Props = {
  currentMode: InspectionModePath;
  cameraId: string | null;
  file: File | null;
  disabled?: boolean;
  onCameraChange: (cameraId: string | null) => void;
  onFileChange: (file: File | null) => void;
};

export function InspectionSourceControl({
  currentMode,
  cameraId,
  file,
  disabled = false,
  onCameraChange,
  onFileChange,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const requiresCamera = MODES_REQUIRING_CAMERA.has(currentMode);
  const { data: cameras = [], isLoading: camerasLoading } = useGetCameras();

  useCameraStatusLive({
    scope: "inspection",
    enabled: requiresCamera,
  });

  const availableCameraOptions = useMemo(() => {
    const serverCameraOptions = cameras
      .filter((camera) => {
        const status = getCameraStatusMeta(camera);
        return camera.is_active && camera.last_status === "online" && !status.isStale;
      })
      .map((camera) => ({
        value: camera.id,
        label: camera.location ? `${camera.name} · ${camera.location}` : camera.name,
      }));

    return [DEVICE_CAMERA_OPTION, ...serverCameraOptions];
  }, [cameras]);

  useEffect(() => {
    if (!requiresCamera || !cameraId) {
      return;
    }

    const selectedStillAvailable =
      isDeviceCameraId(cameraId) ||
      availableCameraOptions.some((option) => option.value === cameraId);

    if (!selectedStillAvailable) {
      onCameraChange(null);
    }
  }, [availableCameraOptions, cameraId, onCameraChange, requiresCamera]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    if (nextFile) {
      onFileChange(nextFile);
    }
    event.currentTarget.value = "";
  };

  if (currentMode === "photo") {
    return (
      <div className={s.root} data-mode="photo">
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          disabled={disabled}
          onChange={handleFileChange}
        />

        <div className={s.sourceMeta}>
          <span><FileImage /> Source</span>
          <b>{file ? file.name : "Фото из файла"}</b>
          <small>{file ? "Файл выбран. Его можно заменить сверху или в viewer." : "Загрузи изображение сверху или перетащи его в viewer."}</small>
        </div>

        <button
          className={s.sourceButton}
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          <Upload /> {file ? "Заменить" : "Выбрать файл"}
        </button>
      </div>
    );
  }

  const isRealtime = currentMode === "realtime";
  const cameraPlaceholder = camerasLoading
    ? "Загрузка камер..."
    : availableCameraOptions.length
      ? "Выберите камеру"
      : "Нет доступных камер";

  return (
    <div className={s.root} data-mode={currentMode}>
      <div className={s.sourceMeta}>
        <span>{isRealtime ? <Video /> : <Camera />} Source</span>
        <b>{isRealtime ? "Realtime camera" : "Camera snapshot"}</b>
        <small>{isRealtime ? "Поток будет использован для live-проверки." : "Будет взят один кадр и отправлен на проверку."}</small>
      </div>

      <div className={s.cameraField}>
        <Select
          noMargin
          placeholder={cameraPlaceholder}
          options={availableCameraOptions}
          value={cameraId ?? ""}
          onChange={(value) => onCameraChange(value || null)}
          disabled={disabled || camerasLoading || availableCameraOptions.length === 0}
        />
      </div>
    </div>
  );
}
