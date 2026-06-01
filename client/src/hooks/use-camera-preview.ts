import { connectCameraPreviewWebRTC } from "@/page-components/cameras/api/camera-preview-webrtc";
import { RefObject, useEffect, useRef, useState } from "react";

export type CameraPreviewState = "idle" | "connecting" | "online" | "reconnecting" | "offline";

export type CameraPreviewError = {
  code: string;
  message: string;
};

export const MAX_AUTO_RECONNECT_ATTEMPTS = 3;

type UseCameraPreviewOptions = {
  autoReconnect?: boolean;
};

export function useCameraPreview(
  cameraId: string | null,
  videoRef: RefObject<HTMLVideoElement | null>,
  fps: number = 20,
  restartToken: number = 0,
  options: UseCameraPreviewOptions = {}
) {
  const [state, setState] = useState<CameraPreviewState>("idle");
  const [error, setError] = useState<CameraPreviewError | null>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const autoReconnect = options.autoReconnect ?? true;

  const cleanupRef = useRef<(() => void) | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;
    let attempt = 0;

    const disposePeer = () => {
      if (cleanupRef.current !== null) {
        const cleanup = cleanupRef.current;
        cleanupRef.current = null;
        cleanup();
      }
    };

    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const cleanup = () => {
      clearReconnectTimer();
      disposePeer();
    };

    if (!cameraId || !videoRef.current) {
      cleanup();
      setState("idle");
      setError(null);
      setReconnectAttempt(0);
      return;
    }

    const scheduleReconnect = (message?: string) => {
      if (cancelled) return;

      clearReconnectTimer();

      if (!autoReconnect) {
        setReconnectAttempt(0);
        setState("offline");
        setError({
          code: "webrtc_preview_disconnected",
          message: message ?? "Не удалось подключиться к камере",
        });
        return;
      }

      attempt += 1;
      if (attempt > MAX_AUTO_RECONNECT_ATTEMPTS) {
        setReconnectAttempt(MAX_AUTO_RECONNECT_ATTEMPTS);
        setState("offline");
        setError({
          code: "webrtc_preview_disconnected",
          message:
            message ??
            "Автопереподключение остановлено. Проверьте камеру и запустите проверку снова.",
        });
        return;
      }

      setReconnectAttempt(attempt);
      setState("reconnecting");
      setError({
        code: "webrtc_preview_disconnected",
        message:
          message ??
          "Соединение с камерой потеряно. Пытаемся переподключиться автоматически.",
      });

      const delay = Math.min(1000 * 2 ** Math.min(attempt - 1, 5), 30_000);

      reconnectTimerRef.current = setTimeout(() => {
        reconnectTimerRef.current = null;
        void connect();
      }, delay);
    };

    const connect = async () => {
      if (cancelled || !videoRef.current || !cameraId) return;

      disposePeer();

      try {
        setState(attempt === 0 ? "connecting" : "reconnecting");
        setError(null);

        cleanupRef.current = await connectCameraPreviewWebRTC({
          cameraId,
          video: videoRef.current,
          fps,
          onDisconnected: (message) => {
            if (cancelled) return;
            disposePeer();
            scheduleReconnect(message);
          },
        });

        if (cancelled) {
          cleanup();
          return;
        }

        attempt = 0;
        setReconnectAttempt(0);
        setState("online");
      } catch (err) {
        if (cancelled) return;

        setError({
          code: "webrtc_preview_failed",
          message: err instanceof Error ? err.message : "Не удалось подключиться к камере",
        });

        scheduleReconnect(err instanceof Error ? err.message : undefined);
      }
    };

    void connect();

    return () => {
      cancelled = true;
      cleanup();
    };
  }, [autoReconnect, cameraId, fps, restartToken, videoRef]);

  return {
    state,
    error,
    isConnected: state === "online",
    reconnectAttempt,
  };
}
