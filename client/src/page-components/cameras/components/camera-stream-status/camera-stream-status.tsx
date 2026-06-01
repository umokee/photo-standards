import type { CameraPreviewState } from "@/hooks/use-camera-preview";
import s from "./camera-stream-status.module.scss";

type Props = {
  state: CameraPreviewState;
  reconnectAttempt: number;
  errorMessage: string | null;
};

export const CameraStreamStatus = ({ state, reconnectAttempt, errorMessage }: Props) => {
  if (state === "online" || state === "idle") {
    return null;
  }

  return (
    <div className={s.root}>
      <div className={s.card}>
        <div className={s.body}>
          <div className={s.title}>{getTitle(state, reconnectAttempt)}</div>
          {errorMessage && state !== "connecting" ? (
            <div className={s.message}>{errorMessage}</div>
          ) : null}
        </div>
      </div>
    </div>
  );
};

function getTitle(state: CameraPreviewState, attempt: number): string {
  switch (state) {
    case "connecting":
      return "Подключение к камере…";
    case "reconnecting":
      return attempt > 0 ? `Переподключение (попытка ${attempt})…` : "Переподключение…";
    case "offline":
      return attempt > 0 ? "Автопереподключение остановлено" : "Камера недоступна";
    default:
      return "";
  }
}
