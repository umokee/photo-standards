import { NavigationBar } from "@/components/layouts/navigation-bar/navigation-bar";
import { Outlet } from "react-router-dom";
import { appNavigation } from "../navigation";

export default function RootLayout() {
  return (
    <>
      <NavigationBar>
        {appNavigation.map(({ to, icon, label }) => (
          <NavigationBar.Link key={to} to={to} icon={icon}>
            {label}
          </NavigationBar.Link>
        ))}
      </NavigationBar>

      <div
        style={{
          display: "flex",
          flex: 1,
          width: "100%",
          minWidth: 0,
          minHeight: 0,
          overflow: "hidden",
        }}
      >
        <Outlet />
      </div>
    </>
  );
}
