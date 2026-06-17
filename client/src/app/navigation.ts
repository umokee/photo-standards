import {
  Camera,
  Database,
  FolderOpen,
  History,
  Home,
  ListChecks,
  Settings,
  TrainFront,
} from "lucide-react";
import { paths } from "./paths";

export const appNavigation = [
  { to: paths.home(), icon: Home, label: "Home", section: "Главная" },
  { to: paths.groups(), icon: Database, label: "Annotate", section: "Эталоны" },
  { to: paths.training(), icon: TrainFront, label: "Train", section: "Модели" },
  { to: paths.inspection(), icon: ListChecks, label: "Inspect", section: "Проверка" },
  { to: paths.inspectionHistory(), icon: History, label: "Runs", section: "История" },
  { to: paths.cameras(), icon: Camera, label: "Cameras", section: "Камеры" },
  { to: paths.settingsSection("system"), icon: Settings, label: "Settings", section: "Система" },
  { to: paths.groups(), icon: FolderOpen, label: "Explore", section: "Обзор" },
] as const;
