import Button from "@/components/ui/button/button";
import QueryState from "@/components/ui/query-state/query-state";
import { CameraStreamStatus } from "@/page-components/cameras/components/camera-stream-status/camera-stream-status";
import {
  useInspectionWorkspace,
  type UseInspectionWorkspaceParams,
} from "@/page-components/inspections/hooks/use-inspection-workspace";
import type { InspectionTaskResult } from "@/types/contracts";
import { Upload } from "lucide-react";
import type { RefObject } from "react";
import { InspectionDropzone } from "../inspection-dropzone/inspection-dropzone";
import { ReferenceThumbnail } from "../reference-thumbnail/reference-thumbnail";
import s from "./inspection-workspace.module.scss";

type Props = UseInspectionWorkspaceParams & {
  activeMatchKey: string | null;
  onActiveMatchChange: (matchKey: string | null) => void;
};

export const InspectionWorkspace = ({
  standardId,
  currentMode,
  file,
  onFileChange,
  cameraId,
  result,
  realtimeSessionId,
  realtimeStatus,
  setRealtimeSessionId,
  taskStatus,
  taskStage,
  taskProgress,
  isLocked,
  selectedClassIds,
  activeMatchKey,
  runDisabled,
  runLabel,
  onRun,
}: Props) => {
  const workspace = useInspectionWorkspace({
    activeMatchKey,
    cameraId,
    currentMode,
    file,
    isLocked,
    onFileChange,
    onRun,
    realtimeSessionId,
    realtimeStatus,
    result,
    runDisabled,
    runLabel,
    selectedClassIds,
    setRealtimeSessionId,
    standardId,
    taskProgress,
    taskStage,
    taskStatus,
  });

  return (
    <div className={s.root}>
      <div className={s.header}>
        <div className={s.info}>
          <span className={s.title}>{workspace.headerTitle}</span>
          <span className={s.subtitle}>{workspace.headerSubtitle}</span>
        </div>

        <div className={s.actions}>
          {currentMode === "photo" && file ? (
            <>
              <input
                ref={workspace.replaceInputRef}
                type="file"
                accept="image/*"
                hidden
                disabled={isLocked}
                onChange={workspace.handleReplaceFile}
              />

              <Button
                variant="ghost"
                icon={Upload}
                disabled={isLocked}
                onClick={() => workspace.replaceInputRef.current?.click()}
              >
                Заменить файл
              </Button>
            </>
          ) : null}

          <Button
            disabled={workspace.primaryActionDisabled}
            onClick={workspace.handlePrimaryAction}
          >
            {workspace.primaryActionLabel}
          </Button>
        </div>
      </div>

      <div className={s.viewer}>
        {renderViewerContent({
          currentMode,
          file,
          isLocked,
          onFileChange,
          displayImageUrl: workspace.displayImageUrl,
          result,
          cameraId,
          snapshotVideoRef: workspace.snapshotVideoRef,
          realtimeVideoRef: workspace.realtimeVideoRef,
          realtimeSessionId,
          realtimeError: workspace.realtimeError,
          deviceCameraError: workspace.deviceCameraError,
          deviceCameraVideoRef: workspace.deviceCameraVideoRef,
          usesDeviceCamera: workspace.usesDeviceCamera,
        })}

        {workspace.referenceImage ? (
          <div className={s.referenceThumbDock}>
            <ReferenceThumbnail
              image={workspace.referenceImage}
              focusedSegmentClassId={workspace.focusedSegmentClassId}
              allowedSegmentClassIds={workspace.allowedSegmentClassIds}
            />
          </div>
        ) : null}

        {currentMode === "snapshot" && !result && cameraId && !workspace.usesDeviceCamera ? (
          <CameraStreamStatus
            state={workspace.snapshotPreviewState}
            reconnectAttempt={workspace.snapshotReconnectAttempt}
            errorMessage={workspace.snapshotPreviewError}
          />
        ) : null}

        {workspace.showTaskOverlay ? (
          <div className={s.overlay}>
            <QueryState isLoading loadingText={taskStage || "Проверка выполняется..."} />
          </div>
        ) : null}
      </div>
    </div>
  );
};

function InspectionImageViewer({ imageUrl }: { imageUrl: string }) {
  return (
    <div className={s.imageViewer}>
      <img className={s.image} src={imageUrl} alt="Результат проверки" />
    </div>
  );
}

function renderViewerContent({
  currentMode,
  file,
  isLocked,
  onFileChange,
  displayImageUrl,
  result,
  cameraId,
  snapshotVideoRef,
  realtimeVideoRef,
  realtimeSessionId,
  realtimeError,
  deviceCameraError,
  deviceCameraVideoRef,
  usesDeviceCamera,
}: {
  currentMode: string;
  file: File | null;
  isLocked: boolean;
  onFileChange: (file: File | null) => void;
  displayImageUrl: string | null;
  result: InspectionTaskResult | null;
  cameraId: string | null;
  snapshotVideoRef: RefObject<HTMLVideoElement | null>;
  realtimeVideoRef: RefObject<HTMLVideoElement | null>;
  realtimeSessionId: string | null;
  realtimeError: string | null;
  deviceCameraError: string | null;
  deviceCameraVideoRef: RefObject<HTMLVideoElement | null>;
  usesDeviceCamera: boolean;
}) {
  if (displayImageUrl) {
    return <InspectionImageViewer imageUrl={displayImageUrl} />;
  }

  if (result) {
    return (
      <QueryState
        isEmpty
        size="block"
        emptyTitle="Нет изображения результата"
        emptyDescription="Сервер завершил проверку, но не сохранил раскрашенное изображение."
      />
    );
  }

  if (currentMode === "photo" && !file) {
    return (
      <div className={s.centerState}>
        <InspectionDropzone disabled={isLocked} onFileSelect={onFileChange} />
      </div>
    );
  }

  if (currentMode === "snapshot" && cameraId && !result) {
    if (usesDeviceCamera) {
      return (
        <>
          <video ref={deviceCameraVideoRef} className={s.video} autoPlay muted playsInline />
          {deviceCameraError ? <div className={s.streamError}>{deviceCameraError}</div> : null}
        </>
      );
    }

    return <video ref={snapshotVideoRef} className={s.video} autoPlay muted playsInline />;
  }

  if (currentMode === "realtime") {
    if (usesDeviceCamera) {
      const isRealtimeStarted = Boolean(realtimeSessionId);

      return (
        <>
          <video
            ref={deviceCameraVideoRef}
            className={isRealtimeStarted ? s.deviceCaptureVideo : s.video}
            autoPlay
            muted
            playsInline
          />

          <video
            ref={realtimeVideoRef}
            className={isRealtimeStarted ? s.video : s.deviceCaptureVideo}
            autoPlay
            muted
            playsInline
          />

          {realtimeError || deviceCameraError ? (
            <div className={s.streamError}>{realtimeError || deviceCameraError}</div>
          ) : null}
        </>
      );
    }

    if (realtimeSessionId) {
      return (
        <>
          <video ref={realtimeVideoRef} className={s.video} autoPlay muted playsInline />
          {realtimeError ? <div className={s.streamError}>{realtimeError}</div> : null}
        </>
      );
    }

    return (
      <div className={s.centerState}>
        <div className={s.emptyCard}>
          <span className={s.emptyTitle}>Live-проверка не запущена</span>
          <span className={s.emptyText}>
            {!cameraId
              ? "Выберите камеру в верхней панели."
              : "После запуска здесь появится видеопоток с результатами контроля"}
          </span>
        </div>
      </div>
    );
  }

  return (
    <QueryState
      isEmpty
      size="block"
      emptyTitle={currentMode === "snapshot" ? "Нет видеопотока" : "Нет изображения"}
      emptyDescription={
        currentMode === "snapshot"
          ? cameraId
            ? "Ожидаем кадры с камеры."
            : "Выберите камеру в верхней панели."
          : "Выберите источник для проверки."
      }
    />
  );
}
