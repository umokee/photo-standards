
import { Outlet } from "react-router-dom";
import s from "./_settings-strict.module.scss";

export function Component() {
  return (
    <div className={s.page}>
      <Outlet />
    </div>
  );
}
