import { Brain, Camera, FolderOpen, History, InspectIcon, Settings } from "lucide-react";
import { paths } from "./paths";

export const appNavigation = [
  { to: paths.groups(), icon: FolderOpen, label: "Группы" },
  { to: paths.training(), icon: Brain, label: "Обучение" },
  { to: paths.inspection(), icon: InspectIcon, label: "Контроль" },
  { to: paths.inspectionHistory(), icon: History, label: "История" },
  { to: paths.cameras(), icon: Camera, label: "Камеры" },
  { to: paths.settingsSection("system"), icon: Settings, label: "Настройки" },
] as const;
