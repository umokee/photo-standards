import { createRoot } from "react-dom/client";
import { initializeTheme } from "@/lib/theme";
import AppProvider from "./app/provider.js";
import "./styles/main.scss";

initializeTheme();

createRoot(document.getElementById("root")!).render(<AppProvider />);
