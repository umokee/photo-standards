import { createRoot } from "react-dom/client";
import AppProvider from "./app/provider.js";
import "./styles/main.scss";

createRoot(document.getElementById("root")!).render(<AppProvider />);
