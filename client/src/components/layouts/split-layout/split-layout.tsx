import useSidebar from "@/hooks/use-sidebar";
import clsx from "clsx";
import React, { createContext, useContext, useLayoutEffect } from "react";
import s from "./split-layout.module.scss";

const InsideRootContext = createContext(false);
const useInsideRoot = (componentName: string) => {
  const insideRoot = useContext(InsideRootContext);

  if (!insideRoot) {
    throw new Error(`${componentName} должен быть использован внутри SplitLayout`);
  }
};

const InsideContentContext = createContext(false);
const useInsideContent = (componentName: string) => {
  const insideContent = useContext(InsideContentContext);

  if (!insideContent) {
    throw new Error(`${componentName} должен быть использован внутри SplitLayout.Content`);
  }
};

const Body = ({ children, bare }: { children: React.ReactNode; bare?: boolean }) => {
  useInsideContent("SplitLayout.Body");
  return <div className={clsx(s.body, bare && s.bare)}>{children}</div>;
};

const Topbar = ({ children }: { children: React.ReactNode }) => {
  useInsideContent("SplitLayout.Topbar");
  return <div className={s.topbar}>{children}</div>;
};

const Bottombar = ({ children }: { children: React.ReactNode }) => {
  useInsideContent("SplitLayout.Bottombar");
  return <div className={s.bottombar}>{children}</div>;
};

const Content = ({ children }: { children: React.ReactNode }) => {
  useInsideRoot("SplitLayout.Content");

  return (
    <InsideContentContext.Provider value={true}>
      <div className={s.content}>{children}</div>
    </InsideContentContext.Provider>
  );
};

const Sidebar = ({ children }: { children: React.ReactNode }) => {
  useInsideRoot("SplitLayout.Sidebar");
  const { open, close, registerSidebar, unregisterSidebar } = useSidebar();

  useLayoutEffect(() => {
    registerSidebar();

    return unregisterSidebar;
  }, [registerSidebar, unregisterSidebar]);

  return (
    <>
      <button
        type="button"
        className={clsx(s.backdrop, open && s.backdropVisible)}
        onClick={close}
        aria-label="Закрыть боковую панель"
        tabIndex={open ? 0 : -1}
      />
      <div className={clsx(s.sidebar, open && s.open)}>{children}</div>
    </>
  );
};

const Panel = ({ children }: { children: React.ReactNode }) => {
  useInsideRoot("SplitLayout.Panel");
  const { open, close, registerSidebar, unregisterSidebar } = useSidebar();

  useLayoutEffect(() => {
    registerSidebar();

    return unregisterSidebar;
  }, [registerSidebar, unregisterSidebar]);

  return (
    <>
      <button
        type="button"
        className={clsx(s.backdrop, open && s.backdropVisible)}
        onClick={close}
        aria-label="Закрыть панель"
        tabIndex={open ? 0 : -1}
      />
      <aside className={clsx(s.panel, open && s.panelOpen)}>
        <button type="button" className={s.panelHandle} onClick={close} aria-label="Скрыть панель">
          <span className={s.panelHandleBar} />
        </button>

        {children}
      </aside>
    </>
  );
};

const Root = ({ children }: { children: React.ReactNode }) => (
  <InsideRootContext.Provider value={true}>
    <div className={s.root}>{children}</div>
  </InsideRootContext.Provider>
);

export const SplitLayout = Object.assign(Root, {
  Content,
  Body,
  Topbar,
  Bottombar,
  Sidebar,
  Panel,
});
