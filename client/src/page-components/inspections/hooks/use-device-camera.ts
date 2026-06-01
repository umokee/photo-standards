import { useEffect, useState, type RefObject } from "react";

export type DeviceCameraState = "idle" | "starting" | "ready" | "error";

type UseDeviceCameraParams = {
  enabled: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
};

export function useDeviceCamera({ enabled, videoRef }: UseDeviceCameraParams) {
  const [state, setState] = useState<DeviceCameraState>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setState("idle");
      setError(null);
      return;
    }

    let cancelled = false;
    let stream: MediaStream | null = null;

    const start = async () => {
      const unavailableMessage = getDeviceCameraUnavailableMessage();

      if (unavailableMessage) {
        setState("error");
        setError(unavailableMessage);
        return;
      }

      setState("starting");
      setError(null);

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280, max: 1920 },
            height: { ideal: 720, max: 1080 },
            frameRate: { ideal: 15, max: 20 },
          },
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        await new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        const video = videoRef.current;

        if (!video) {
          throw new Error("Видеоэлемент камеры ещё не готов");
        }

        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;

        await video.play();

        setState("ready");
      } catch (err) {
        if (cancelled) {
          return;
        }

        stream?.getTracks().forEach((track) => track.stop());
        stream = null;

        setState("error");
        setError(normalizeDeviceCameraError(err));
      }
    };

    void start();

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((track) => track.stop());

      const video = videoRef.current;
      if (video?.srcObject === stream) {
        video.srcObject = null;
      }
    };
  }, [enabled, videoRef]);

  return { state, error };
}

function getDeviceCameraUnavailableMessage(): string | null {
  if (!window.isSecureContext) {
    return "Доступ к камере доступен только через HTTPS или localhost. Откройте приложение по HTTPS.";
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    return "Браузер не поддерживает доступ к камере через getUserMedia";
  }

  return null;
}

function normalizeDeviceCameraError(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError") {
      return "Доступ к камере запрещён. Разрешите доступ в настройках браузера.";
    }

    if (error.name === "NotFoundError") {
      return "Камера устройства не найдена.";
    }

    if (error.name === "NotReadableError") {
      return "Камера уже используется другим приложением или браузер не может её открыть.";
    }

    if (error.name === "OverconstrainedError") {
      return "Камера не поддерживает запрошенные параметры видео.";
    }

    if (error.message) {
      return error.message;
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "Не удалось открыть камеру устройства";
}
