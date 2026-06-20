export type AppTheme = "light" | "dark";

export const THEME_STORAGE_KEY = "vc-theme";
export const THEME_CHANGE_EVENT = "vc-theme-change";

const THEMES = new Set<AppTheme>(["light", "dark"]);

export function isAppTheme(value: unknown): value is AppTheme {
  return typeof value === "string" && THEMES.has(value as AppTheme);
}

export function getSystemTheme(): AppTheme {
  if (typeof window === "undefined") return "light";

  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function readStoredTheme(): AppTheme | null {
  if (typeof window === "undefined") return null;

  try {
    const value = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isAppTheme(value) ? value : null;
  } catch {
    return null;
  }
}

export function getCurrentTheme(): AppTheme {
  if (typeof document === "undefined") return "light";

  const current = document.documentElement.dataset.theme;
  if (isAppTheme(current)) return current;

  return readStoredTheme() ?? getSystemTheme();
}

export function applyTheme(theme: AppTheme): void {
  if (typeof document === "undefined") return;

  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
}

export function initializeTheme(): AppTheme {
  const theme = readStoredTheme() ?? getSystemTheme();
  applyTheme(theme);
  return theme;
}

export function setAppTheme(theme: AppTheme): void {
  applyTheme(theme);

  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
  }

  window.dispatchEvent(new CustomEvent<AppTheme>(THEME_CHANGE_EVENT, { detail: theme }));
}

export function subscribeTheme(listener: (theme: AppTheme) => void): () => void {
  const handleThemeChange = (event: Event) => {
    const nextTheme = event instanceof CustomEvent && isAppTheme(event.detail) ? event.detail : getCurrentTheme();
    listener(nextTheme);
  };

  const handleStorage = (event: StorageEvent) => {
    if (event.key !== THEME_STORAGE_KEY) return;
    listener(readStoredTheme() ?? getSystemTheme());
  };

  window.addEventListener(THEME_CHANGE_EVENT, handleThemeChange);
  window.addEventListener("storage", handleStorage);

  return () => {
    window.removeEventListener(THEME_CHANGE_EVENT, handleThemeChange);
    window.removeEventListener("storage", handleStorage);
  };
}
