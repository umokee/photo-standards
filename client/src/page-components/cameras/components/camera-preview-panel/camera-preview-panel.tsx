import {
  useCameraPreview,
  type CameraPreviewState,
} from "@/hooks/use-camera-preview";
import type { Camera } from "@/types/contracts";
import clsx from "clsx";
import { useEffect, useRef, useState } from "react";
import s from "./camera-preview-panel.module.scss";

type Props = {
  camera: Camera;
};

export const CameraPreviewPanel = ({ camera }: Props) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [previewStarted, setPreviewStarted] = useState(false);
  const [previewRestartToken, setPreviewRestartToken] = useState(0);

  const { state, error } = useCameraPreview(
    previewStarted ? camera.id : null,
    videoRef,
    20,
    previewRestartToken,
    { autoReconnect: false }
  );

  useEffect(() => {
    setPreviewStarted(false);
    setPreviewRestartToken(0);
  }, [camera.id]);

  const isPreviewOnline = previewStarted && state === "online";
  const stageActionTitle = getStageActionTitle({
    previewStarted,
    state,
  });
  const stageDiagnostic = getStageDiagnostic({
    previewStarted,
    state,
    streamErrorMessage: error?.message ?? null,
  });

  const handleOpenStream = () => {
    if (state === "online") {
      return;
    }

    setPreviewStarted(true);
    setPreviewRestartToken((current) => current + 1);
  };

  return (
    <div className={s.panel}>
      <div className={s.previewViewport}>
        {previewStarted ? (
          <video ref={videoRef} className={s.previewVideo} autoPlay muted playsInline />
        ) : null}

        {!isPreviewOnline ? (
          <button
            type="button"
            className={clsx(s.previewAction, s.previewActionCentered)}
            onClick={handleOpenStream}
            disabled={state === "connecting"}
            aria-label={stageActionTitle}
          >
            <span className={s.previewActionLabel}>{stageActionTitle}</span>
          </button>
        ) : null}
      </div>

      {stageDiagnostic ? (
        <div className={clsx(s.statusPanel, stageDiagnostic.isAlert && s.statusPanelAlert)}>
          <div className={s.statusHeader}>
            <span className={s.statusTitle}>{stageDiagnostic.title}</span>
          </div>
          <div className={s.statusMessage}>{stageDiagnostic.message}</div>
        </div>
      ) : null}
    </div>
  );
};

function getStageActionTitle({
  previewStarted,
  state,
}: {
  previewStarted: boolean;
  state: CameraPreviewState;
}) {
  if (!previewStarted) {
    return "Открыть поток";
  }

  if (state === "idle" || state === "connecting") {
    return "Открываем поток";
  }

  return "Повторить попытку";
}

function getStageDiagnostic({
  previewStarted,
  state,
  streamErrorMessage,
}: {
  previewStarted: boolean;
  state: CameraPreviewState;
  streamErrorMessage: string | null;
}) {
  if (state === "online") {
    return null;
  }

  if (state === "offline" && streamErrorMessage) {
    return {
      title: "Поток недоступен",
      message: streamErrorMessage,
      badgeLabel: "Недоступна",
      badgeType: "danger" as const,
      isAlert: true,
    };
  }

  if (state === "idle" || state === "connecting") {
    return null;
  }

  if (!previewStarted) {
    return null;
  }

  return null;
}
