import { queryClient } from "@/lib/query-client";
import { getCameraQueryOptions } from "@/page-components/cameras/api/get-camera";
import { getCamerasQueryOptions } from "@/page-components/cameras/api/get-cameras";
import { RouteObject } from "react-router-dom";
import { paths } from "../../paths";
import { requireParam } from "../../route-params";
import RouteError from "../route-error";
import RouteLoadingFallback from "../route-loading-fallback";

const CamerasLayoutRoute = () => import("./_cameras-layout");
const CamerasIndexRoute = () => import("./_cameras-index");
const CameraDetailRoute = () => import("./_camera-detail");

export const camerasRoute: RouteObject = {
  path: paths.cameras(),
  lazy: CamerasLayoutRoute,
  errorElement: <RouteError />,
  hydrateFallbackElement: <RouteLoadingFallback />,
  loader: async () => {
    await queryClient.ensureQueryData(getCamerasQueryOptions());
  },
  children: [
    {
      index: true,
      lazy: CamerasIndexRoute,
    },
    {
      path: ":cameraId",
      lazy: CameraDetailRoute,
      loader: async ({ params }) => {
        const cameraId = requireParam(params.cameraId, paths.cameras());

        await queryClient.ensureQueryData(getCameraQueryOptions(cameraId));

        return { cameraId };
      },
    },
  ],
};
