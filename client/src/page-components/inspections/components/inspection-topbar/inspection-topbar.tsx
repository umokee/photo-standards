import { inspectionModePaths, type InspectionModePath } from "@/app/paths";
import Select from "@/components/ui/select/select";
import { useInspectionModeOptions } from "@/constants";
import { useGetCameras } from "@/page-components/cameras/api/get-cameras";
import { useCameraStatusLive } from "@/page-components/cameras/hooks/use-camera-status-live";
import { getCameraStatusMeta } from "@/page-components/cameras/lib/camera-view";
import { getGroupQueryOptions } from "@/page-components/groups/api/get-group";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { useQuery } from "@tanstack/react-query";
import { memo, useEffect, useMemo } from "react";
import { DEVICE_CAMERA_OPTION, isDeviceCameraId } from "../../lib/device-camera";
import s from "./inspection-topbar.module.scss";

const MODES_REQUIRING_CAMERA = new Set<InspectionModePath>(["snapshot", "realtime"]);

interface Props {
  currentMode: InspectionModePath;
  groupId: string | null;
  standardId: string | null;
  cameraId: string | null;
  onModeChange: (mode: InspectionModePath) => void;
  onGroupChange: (groupId: string | null) => void;
  onStandardChange: (standardId: string | null) => void;
  onCameraChange: (cameraId: string | null) => void;
}

export const InspectionTopbar = ({
  currentMode,
  groupId,
  standardId,
  cameraId,
  onModeChange,
  onGroupChange,
  onStandardChange,
  onCameraChange,
}: Props) => {
  const { data: groups = [] } = useGetGroups();
  const { data: cameras = [], isLoading: camerasLoading } = useGetCameras();

  const modeOptions = useInspectionModeOptions();
  const requiresCamera = MODES_REQUIRING_CAMERA.has(currentMode);

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

  const groupOptions = groups.map((group) => ({
    value: group.id,
    label: group.name,
  }));

  const cameraPlaceholder = camerasLoading ? "Загрузка камер..." : "Выберите камеру";

  return (
    <div className={s.root}>
      <div className={s.field}>
        <Select
          noMargin
          options={modeOptions}
          value={currentMode}
          onChange={(value) => {
            if (isInspectionModePath(value)) {
              onModeChange(value);
            }
          }}
        />
      </div>

      <div className={s.field}>
        <Select
          noMargin
          placeholder="Выберите группу"
          options={groupOptions}
          value={groupId ?? ""}
          onChange={(value) => onGroupChange(value || null)}
        />
      </div>

      {groupId ? (
        <InspectionStandardSelect
          groupId={groupId}
          value={standardId ?? ""}
          onChange={(value) => onStandardChange(value || null)}
        />
      ) : (
        <div className={s.field}>
          <Select
            noMargin
            disabled
            placeholder="Сначала выберите группу"
            options={[]}
            value=""
            onChange={() => {}}
          />
        </div>
      )}

      {requiresCamera ? (
        <div className={s.field}>
          <Select
            noMargin
            placeholder={cameraPlaceholder}
            options={availableCameraOptions}
            value={cameraId ?? ""}
            onChange={(value) => onCameraChange(value || null)}
            disabled={availableCameraOptions.length === 0}
          />
        </div>
      ) : (
        <div className={s.field}>
          <Select
            noMargin
            disabled
            placeholder="Камера не нужна"
            options={[]}
            value=""
            onChange={() => {}}
          />
        </div>
      )}
    </div>
  );
};

const InspectionStandardSelect = memo(function InspectionStandardSelect({
  groupId,
  value,
  onChange,
}: {
  groupId: string;
  value: string;
  onChange: (standardId: string) => void;
}) {
  const {
    data: group,
    isPending,
    isError,
  } = useQuery({
    ...getGroupQueryOptions(groupId),
    enabled: Boolean(groupId),
  });

  const options =
    group?.standards.map((standard) => ({
      value: standard.id,
      label: standard.name,
    })) ?? [];

  const placeholder = isPending
    ? "Загрузка эталонов..."
    : isError
      ? "Не удалось загрузить эталоны"
      : options.length
        ? "Выберите эталон"
        : "В группе нет эталонов";

  return (
    <div className={s.field}>
      <Select
        noMargin
        placeholder={placeholder}
        options={options}
        value={value}
        onChange={onChange}
        disabled={isPending || isError || options.length === 0}
      />
    </div>
  );
});

function isInspectionModePath(value: string): value is InspectionModePath {
  return inspectionModePaths.includes(value as InspectionModePath);
}
