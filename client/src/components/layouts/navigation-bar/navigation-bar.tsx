import useSidebar from "@/hooks/use-sidebar";
import clsx from "clsx";
import { LucideIcon, Menu } from "lucide-react";
import { createContext, ReactNode, useContext } from "react";
import { NavLink } from "react-router-dom";
import s from "./navigation-bar.module.scss";

type LinkProps = {
  to: string;
  icon: LucideIcon;
  children: ReactNode;
};

const InsideRootContext = createContext(false);
const useInsideRoot = (componentName: string) => {
  const insideRoot = useContext(InsideRootContext);

  if (!insideRoot) {
    throw new Error(`${componentName} должен быть использован внутри NavigationBar`);
  }
};

const Link = ({ to, icon: Icon, children }: LinkProps) => {
  useInsideRoot("NavigationBar.Link");

  return (
    <NavLink to={to} className={({ isActive }) => clsx(s.link, isActive && s.linkActive)}>
      <Icon className={s.linkIcon} />
      {children}
    </NavLink>
  );
};

const Root = ({ children }: { children: ReactNode }) => {
  const { hasSidebar, open, toggle } = useSidebar();

  return (
    <InsideRootContext.Provider value={true}>
      <header className={s.root}>
        <div className={s.inner}>
          {hasSidebar && (
            <button
              type="button"
              className={s.menuButton}
              onClick={toggle}
              aria-label={open ? "Скрыть панель" : "Открыть панель"}
              aria-pressed={open}
            >
              <Menu />
            </button>
          )}

          <nav className={s.links} aria-label="Основная навигация">
            {children}
          </nav>
        </div>
      </header>
    </InsideRootContext.Provider>
  );
};

export const NavigationBar = Object.assign(Root, { Link });
