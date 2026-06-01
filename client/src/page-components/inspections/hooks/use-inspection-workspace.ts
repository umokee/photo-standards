import { useCameraPreview } from "@/hooks/use-camera-preview";
import { useStopRealtimeSession } from "@/page-components/inspections/api/realtime-session";
import { connectRealtimeInspectionWebRTC } from "@/page-components/inspections/api/realtime-webrtc";
import {
  buildRunInspectionPayload,
  useRunInspection,
} from "@/page-components/inspections/api/run-inspection";
import { getImageQueryOptions } from "@/page-components/standards/api/get-image";
import { getStandardQueryOptions } from "@/page-components/standards/api/get-standard";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import type { InspectionRealtimeStatus, InspectionTaskResult } from "@/types/contracts";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { captureVideoFrameAsFile, DEVICE_CAMERA_ID, isDeviceCameraId } from "../lib/device-camera";
import type { InspectionRunOverrides } from "../lib/inspection-context";
import { useDeviceCamera } from "./use-device-camera";

type RealtimeStatusLike = InspectionRealtimeStatus | null | undefined;

export type UseInspectionWorkspaceParams = {
  standardId: string | null;
  currentMode: string;
  file: File | null;
  onFileChange: (file: File | null) => void;
  cameraId: string | null;
  result: InspectionTaskResult | null;
  realtimeSessionId: string | null;
  realtimeStatus: InspectionRealtimeStatus | null;
  setRealtimeSessionId: (sessionId: string | null) => void;
  taskStatus: string | null;
  taskStage: string | null;
  taskProgress: number | null;
  isLocked: boolean;
  selectedClassIds: string[];
  activeMatchKey: string | null;
  runDisabled: boolean;
  runLabel: string;
  onRun: (overrides?: InspectionRunOverrides) => void;
};

export function useInspectionWorkspace({
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
}: UseInspectionWorkspaceParams) {
  const objectUrl = useObjectUrl(file);

  const replaceInputRef = useRef<HTMLInputElement | null>(null);
  const snapshotVideoRef = useRef<HTMLVideoElement | null>(null);
  const realtimeVideoRef = useRef<HTMLVideoElement | null>(null);
  const deviceCameraVideoRef = useRef<HTMLVideoElement | null>(null);

  const [realtimeError, setRealtimeError] = useState<string | null>(null);
  const [deviceSnapshotError, setDeviceSnapshotError] = useState<string | null>(null);

  const usesDeviceCamera = isDeviceCameraId(cameraId);
  const deviceCameraEnabled =
    usesDeviceCamera && (currentMode === "snapshot" || currentMode === "realtime") && !result;

  const deviceCamera = useDeviceCamera({
    enabled: deviceCameraEnabled,
    videoRef: deviceCameraVideoRef,
  });

  const realtimeSessionIdRef = useRef<string | null>(null);
  const realtimeCleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    realtimeSessionIdRef.current = realtimeSessionId;
  }, [realtimeSessionId]);

  const { data: standard } = useQuery({
    ...getStandardQueryOptions(standardId ?? ""),
    enabled: Boolean(standardId),
  });

  const referenceImageId = standard?.stats.reference_image_id ?? null;

  const { data: referenceImage } = useQuery({
    ...getImageQueryOptions(referenceImageId ?? ""),
    enabled: Boolean(referenceImageId),
  });

  const {
    state: snapshotPreviewState,
    error: snapshotPreviewError,
    reconnectAttempt: snapshotReconnectAttempt,
  } = useCameraPreview(
    currentMode === "snapshot" && Boolean(cameraId) && !usesDeviceCamera && !result
      ? cameraId
      : null,
    snapshotVideoRef
  );

  const realtimeStartMutation = useRunInspection();
  const realtimeStopMutation = useStopRealtimeSession();

  useEffect(() => {
    if (!realtimeSessionId || !realtimeVideoRef.current) {
      return;
    }

    if (usesDeviceCamera && deviceCamera.state !== "ready") {
      return;
    }

    let cancelled = false;
    setRealtimeError(null);

    const localStream =
      usesDeviceCamera && deviceCameraVideoRef.current?.srcObject instanceof MediaStream
        ? deviceCameraVideoRef.current.srcObject
        : undefined;

    if (usesDeviceCamera && !localStream) {
      setRealtimeError("Камера устройства открыта, но поток ещё не готов для WebRTC.");
      return;
    }

    connectRealtimeInspectionWebRTC({
      sessionId: realtimeSessionId,
      video: realtimeVideoRef.current,
      fps: usesDeviceCamera ? 15 : 20,
      localStream,
      receiveVideo: true,
    })
      .then((cleanup) => {
        if (cancelled) {
          cleanup();
          return;
        }

        realtimeCleanupRef.current = cleanup;
      })
      .catch((error) => {
        setRealtimeError(
          error instanceof Error ? error.message : "Не удалось подключить видеопоток"
        );
      });

    return () => {
      cancelled = true;
      realtimeCleanupRef.current?.();
      realtimeCleanupRef.current = null;
    };
  }, [realtimeSessionId, usesDeviceCamera, deviceCamera.state]);

  useEffect(() => {
    return () => {
      realtimeCleanupRef.current?.();
      realtimeCleanupRef.current = null;
    };
  }, []);

  const resultImageUrl = result?.result_image_path ? `/storage/${result.result_image_path}` : null;

  const displayImageUrl = useMemo(() => {
    if (resultImageUrl) {
      return resultImageUrl;
    }

    if (currentMode === "photo" && !result) {
      return objectUrl;
    }

    return null;
  }, [currentMode, objectUrl, result, resultImageUrl]);

  const focusedSegmentClassId = useMemo(() => {
    return getFocusedSegmentClassId(
      currentMode === "realtime" ? realtimeStatus : result,
      activeMatchKey
    );
  }, [activeMatchKey, currentMode, realtimeStatus, result]);

  const allowedSegmentClassIds = useMemo(() => {
    const currentInspectionResult = currentMode === "realtime" ? realtimeStatus : result;

    if (currentInspectionResult?.details.length) {
      return new Set(
        currentInspectionResult.details
          .map((detail) => detail.segment_class_id)
          .filter((id): id is string => Boolean(id))
      );
    }

    if (selectedClassIds.length > 0) {
      return new Set(selectedClassIds);
    }

    return new Set(referenceImage?.segment_classes.map((item) => item.id) ?? []);
  }, [currentMode, referenceImage?.segment_classes, realtimeStatus, result, selectedClassIds]);

  const showTaskOverlay =
    currentMode !== "realtime" &&
    ((currentMode === "photo" && Boolean(file)) ||
      (currentMode === "snapshot" && Boolean(cameraId))) &&
    isActiveTaskStatus(taskStatus) &&
    !result;

  const canStartRealtime =
    Boolean(standardId) &&
    Boolean(cameraId) &&
    selectedClassIds.length > 0 &&
    !isLocked &&
    !realtimeSessionId &&
    !realtimeStartMutation.isPending;

  const headerTitle = getWorkspaceTitle({
    cameraId,
    currentMode,
    file,
    realtimeSessionId,
  });

  const deviceCameraErrorText = deviceSnapshotError || deviceCamera.error;

  const headerSubtitle = getWorkspaceSubtitle({
    cameraId,
    currentMode,
    realtimeStatus,
    result,
    selectedClassIds,
    standardId,
    taskProgress,
    taskStage,
    usesDeviceCamera,
    deviceCameraState: deviceCamera.state,
    deviceCameraError: deviceCameraErrorText,
  });

  const handleReplaceFile = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    onFileChange(nextFile);
    event.target.value = "";
  };

  const startRealtime = async () => {
    const payloadResult = buildRunInspectionPayload({
      standard_id: standardId,
      selected_segment_class_ids: selectedClassIds,
      mode: "realtime",
      camera_id: usesDeviceCamera ? DEVICE_CAMERA_ID : cameraId,
    });

    if ("errors" in payloadResult) {
      const firstKey = Object.keys(payloadResult.errors)[0];
      const firstError = firstKey ? payloadResult.errors[firstKey] : undefined;

      setRealtimeError(firstError ?? "Не удалось подготовить запуск live-проверки");
      return;
    }

    setRealtimeError(null);

    try {
      const response = await realtimeStartMutation.mutateAsync(payloadResult.data);

      if (response.kind !== "session" || !response.session_id) {
        throw new Error("Сервер не вернул realtime session_id");
      }

      setRealtimeSessionId(response.session_id);
    } catch (err) {
      setRealtimeError(err instanceof Error ? err.message : "Не удалось запустить live-проверку");
    }
  };

  const stopRealtime = async () => {
    const sessionId = realtimeSessionIdRef.current;

    if (!sessionId) {
      return;
    }

    realtimeCleanupRef.current?.();
    realtimeCleanupRef.current = null;

    setRealtimeSessionId(null);
    setRealtimeError(null);

    try {
      await realtimeStopMutation.mutateAsync(sessionId);
    } catch {}
  };

  const handlePrimaryAction = async () => {
    setDeviceSnapshotError(null);

    if (currentMode === "snapshot" && usesDeviceCamera) {
      if (deviceCamera.state !== "ready") {
        setDeviceSnapshotError(
          deviceCamera.error ||
            "Камера устройства ещё не готова. Проверьте HTTPS и разрешение камеры."
        );
        return;
      }

      try {
        const image = await captureVideoFrameAsFile(
          deviceCameraVideoRef.current,
          "device-snapshot.jpg"
        );
        onRun({ image, cameraId: null });
      } catch (err) {
        setDeviceSnapshotError(
          err instanceof Error ? err.message : "Не удалось получить кадр камеры"
        );
      }
      return;
    }

    if (currentMode === "realtime") {
      if (realtimeSessionId) {
        void stopRealtime();
        return;
      }

      if (usesDeviceCamera) {
        if (deviceCamera.state !== "ready") {
          setRealtimeError(
            deviceCamera.error ||
              "Камера устройства ещё не готова. Проверьте HTTPS и разрешение камеры."
          );
          return;
        }

        const stream = deviceCameraVideoRef.current?.srcObject;

        if (!(stream instanceof MediaStream) || stream.getVideoTracks().length === 0) {
          setRealtimeError("Камера устройства открыта, но видеопоток ещё не готов.");
          return;
        }

        if (!stream.getVideoTracks().some((track) => track.readyState === "live")) {
          setRealtimeError("Видеопоток камеры устройства уже остановлен.");
          return;
        }
      }

      void startRealtime();
      return;
    }

    onRun();
  };

  const primaryActionLabel =
    currentMode === "realtime"
      ? realtimeSessionId
        ? realtimeStopMutation.isPending
          ? "Остановка..."
          : "Остановить"
        : realtimeStartMutation.isPending
          ? "Запуск..."
          : "Начать live-проверку"
      : runLabel;

  const primaryActionDisabled =
    currentMode === "realtime"
      ? realtimeSessionId
        ? realtimeStopMutation.isPending
        : !canStartRealtime
      : runDisabled;

  return {
    allowedSegmentClassIds,
    canStartRealtime,
    displayImageUrl,
    focusedSegmentClassId,
    handlePrimaryAction,
    handleReplaceFile,
    headerSubtitle,
    headerTitle,
    primaryActionDisabled,
    primaryActionLabel,
    realtimeError,
    realtimeSessionId,
    realtimeStartPending: realtimeStartMutation.isPending,
    referenceImage,
    replaceInputRef,
    showTaskOverlay,
    snapshotPreviewError: usesDeviceCamera
      ? deviceSnapshotError || deviceCamera.error
      : (snapshotPreviewError?.message ?? null),
    snapshotPreviewState,
    snapshotReconnectAttempt,
    snapshotVideoRef,
    startRealtime,
    realtimeVideoRef,
    deviceCameraError: deviceCameraErrorText,
    deviceCameraState: deviceCamera.state,
    deviceCameraVideoRef,
    usesDeviceCamera,
  };
}

function useObjectUrl(file: File | null) {
  const url = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  useEffect(() => {
    return () => {
      if (url) {
        URL.revokeObjectURL(url);
      }
    };
  }, [url]);

  return url;
}

function getFocusedSegmentClassId(
  result: InspectionTaskResult | InspectionRealtimeStatus | null | undefined,
  activeMatchKey: string | null
): string | null {
  if (!result || activeMatchKey === null) {
    return null;
  }

  const idx = Number(activeMatchKey);

  if (Number.isNaN(idx)) {
    return null;
  }

  return result.details[idx]?.segment_class_id ?? null;
}

function getWorkspaceTitle({
  cameraId,
  currentMode,
  file,
  realtimeSessionId,
}: {
  cameraId: string | null;
  currentMode: string;
  file: File | null;
  realtimeSessionId: string | null;
}) {
  if (currentMode === "photo") {
    return file?.name || "Файл не выбран";
  }

  if (currentMode === "snapshot") {
    return cameraId ? "Предпросмотр камеры" : "Камера не выбрана";
  }

  if (currentMode === "realtime") {
    return realtimeSessionId ? "Live-проверка" : "Live-проверка не запущена";
  }

  return "Проверка";
}

function getWorkspaceSubtitle({
  cameraId,
  currentMode,
  realtimeStatus,
  result,
  selectedClassIds,
  standardId,
  taskProgress,
  taskStage,
  usesDeviceCamera,
  deviceCameraState,
  deviceCameraError,
}: {
  cameraId: string | null;
  currentMode: string;
  realtimeStatus: RealtimeStatusLike;
  result: InspectionTaskResult | null;
  selectedClassIds: string[];
  standardId: string | null;
  taskProgress: number | null;
  taskStage: string | null;
  usesDeviceCamera: boolean;
  deviceCameraState: "idle" | "starting" | "ready" | "error";
  deviceCameraError: string | null;
}) {
  if (result) {
    return `Результат: ${result.matched} / ${result.total}`;
  }

  if (taskStage) {
    return taskProgress == null ? taskStage : `${taskStage} · ${taskProgress}%`;
  }

  if (currentMode === "photo") {
    return "Загрузите изображение, затем запустите проверку";
  }

  if (currentMode === "snapshot") {
    if (!cameraId) {
      return "Выберите камеру, чтобы увидеть предпросмотр";
    }

    if (usesDeviceCamera) {
      if (deviceCameraState === "starting") {
        return "Запрашиваем доступ к камере устройства...";
      }

      if (deviceCameraState === "ready") {
        return "Камера устройства готова для снимка и проверки";
      }

      if (deviceCameraState === "error") {
        return deviceCameraError ?? "Не удалось открыть камеру устройства";
      }

      return "Ожидаем камеру устройства...";
    }

    return "Камера готова для снимка и проверки";
  }

  if (currentMode === "realtime") {
    if (realtimeStatus) {
      if (realtimeStatus.state === "warming_up") {
        return "Подготовка модели и видеопотока...";
      }

      return `${realtimeStatus.matched ?? 0} / ${realtimeStatus.total ?? 0} · ${
        realtimeStatus.passed ? "пройдено" : "не пройдено"
      }`;
    }

    if (!standardId) {
      return "Выберите эталон.";
    }

    if (!cameraId) {
      return "Выберите камеру.";
    }

    if (selectedClassIds.length === 0) {
      return "Выберите классы справа.";
    }

    return "Запустите проверку — кадры будут обрабатываться в реальном времени.";
  }

  return "";
}
