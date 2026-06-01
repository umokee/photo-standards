import { createContext, useCallback, useMemo, useState } from "react";

interface SidebarContextValue {
  hasSidebar: boolean;
  open: boolean;
  toggle: () => void;
  close: () => void;
  registerSidebar: () => void;
  unregisterSidebar: () => void;
}

export const SidebarContext = createContext<SidebarContextValue>({
  hasSidebar: false,
  open: false,
  toggle: () => {},
  close: () => {},
  registerSidebar: () => {},
  unregisterSidebar: () => {},
});

export function SidebarProvider({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const [sidebarCount, setSidebarCount] = useState(0);

  const toggle = useCallback(() => {
    setOpen((isOpen) => !isOpen);
  }, []);

  const close = useCallback(() => {
    setOpen(false);
  }, []);

  const registerSidebar = useCallback(() => {
    setSidebarCount((count) => count + 1);
  }, []);

  const unregisterSidebar = useCallback(() => {
    setSidebarCount((count) => Math.max(0, count - 1));
    setOpen(false);
  }, []);

  const value = useMemo(
    () => ({
      hasSidebar: sidebarCount > 0,
      open,
      toggle,
      close,
      registerSidebar,
      unregisterSidebar,
    }),
    [close, open, registerSidebar, sidebarCount, toggle, unregisterSidebar]
  );

  return <SidebarContext.Provider value={value}>{children}</SidebarContext.Provider>;
}
