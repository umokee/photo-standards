import { PlatformShell } from "@/components/layouts/platform-shell/platform-shell";
import { appNavigation } from "../navigation";

export default function RootLayout() {
  return <PlatformShell navigation={appNavigation} />;
}
