import { queryClient } from "@/lib/query-client";
import { RouteObject, redirect } from "react-router-dom";
import { paths } from "../../paths";
import RouteError from "../route-error";
import RouteLoadingFallback from "../route-loading-fallback";

const SettingsLayoutRoute = () => import("./_settings-layout");
const SettingsSystemRoute = () => import("./_settings-system");

export const settingsRoute: RouteObject = {
  path: paths.settings(),
  lazy: SettingsLayoutRoute,
  errorElement: <RouteError />,
  hydrateFallbackElement: <RouteLoadingFallback />,
  children: [
    {
      index: true,
      loader: async () => {
        throw redirect(paths.settingsSection("system"));
      },
    },
    {
      path: "system",
      lazy: SettingsSystemRoute,
      loader: async () => {
        const { getSystemStatsQueryOptions } = await import(
          "@/page-components/settings/api/get-system-stats"
        );
        await queryClient.ensureQueryData(getSystemStatsQueryOptions());
      },
    },
  ],
};
