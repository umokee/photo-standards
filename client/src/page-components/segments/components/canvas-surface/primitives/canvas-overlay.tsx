import clsx from "clsx";
import s from "../canvas-surface.module.scss";

type Props = {
  isDrawing: boolean;
  modeLabel: string;
  hintText: string;
};

export function CanvasOverlay({ isDrawing, modeLabel, hintText }: Props) {
  return (
    <>
      <div className={clsx(s.modeBadge, isDrawing && s.modeBadgeActive)}>{modeLabel}</div>
      <div className={s.hint}>{hintText}</div>
    </>
  );
}
