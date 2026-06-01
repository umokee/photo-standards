import { queryClient } from "@/lib/query-client";
import { getConstantsQueryOptions } from "@/page-components/meta/get-constants";
import { createBrowserRouter, redirect } from "react-router-dom";
import { paths } from "./paths";
import { camerasRoute } from "./routes/cameras/routes";
import { groupsRoute } from "./routes/groups/routes";
import { imagesRoute } from "./routes/images/routes";
import { inspectionHistoryRoute } from "./routes/inspection-history/routes";
import { inspectionModeRoute, inspectionRedirectRoute } from "./routes/inspection/routes";
import RootLayout from "./routes/root";
import RouteError from "./routes/route-error";
import RouteLoadingFallback from "./routes/route-loading-fallback";
import { settingsRoute } from "./routes/settings/routes";
import { trainingRoute } from "./routes/training/routes";

export const router = createBrowserRouter([
  {
    element: <RootLayout />,
    errorElement: <RouteError />,
    hydrateFallbackElement: <RouteLoadingFallback />,
    loader: async () => {
      await queryClient.ensureQueryData(getConstantsQueryOptions());
    },
    children: [
      {
        path: paths.home(),
        loader: async () => {
          throw redirect(paths.groups());
        },
      },
      groupsRoute,
      imagesRoute,
      trainingRoute,
      inspectionRedirectRoute,
      inspectionModeRoute,
      inspectionHistoryRoute,
      camerasRoute,
      settingsRoute,
    ],
  },
]);
