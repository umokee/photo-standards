import { createRoot } from "react-dom/client";
import AppProvider from "./app/provider.js";
import "./styles/main.scss";

const THEME_STORAGE_KEY = "visionqc-theme";
const THEME_VALUES = new Set(["light", "dark"]);

try {
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  const systemTheme = window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  const initialTheme = THEME_VALUES.has(storedTheme ?? "") ? storedTheme : systemTheme;
  document.documentElement.dataset.theme = initialTheme ?? "light";
} catch {
  document.documentElement.dataset.theme = "light";
}


createRoot(document.getElementById("root")!).render(<AppProvider />);
