import { Outlet } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  return (
    <div className={p.page}>
      <Outlet />
    </div>
  );
}
