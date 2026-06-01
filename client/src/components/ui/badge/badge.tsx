import clsx from "clsx";
import s from "./badge.module.scss";

interface Props {
  type?: "info" | "success" | "warning" | "danger";
  colorDot?: number | null;
  children: React.ReactNode;
}

export const Badge = ({ type = "info", colorDot, children }: Props) => {
  const hasColorDot = colorDot !== undefined && colorDot !== null;

  return (
    <div className={clsx(s.root, s[type])}>
      {hasColorDot && (
        <span className={s.colorDot} style={{ background: `hsl(${colorDot}, 65%, 55%)` }} />
      )}
      {children}
    </div>
  );
};
