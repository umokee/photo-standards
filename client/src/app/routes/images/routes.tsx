import { queryClient } from "@/lib/query-client";
import { getGroupQueryOptions } from "@/page-components/groups/api/get-group";
import { getImageQueryOptions } from "@/page-components/standards/api/get-image";
import { getStandardQueryOptions } from "@/page-components/standards/api/get-standard";
import { RouteObject } from "react-router-dom";
import { paths } from "../../paths";
import { requireParam } from "../../route-params";
import RouteError from "../route-error";
import RouteLoadingFallback from "../route-loading-fallback";

const ImagesRoute = () => import("./_images");

export const imagesRoute: RouteObject = {
  path: "/groups/:groupId/standards/:standardId/images/:imageId",
  lazy: ImagesRoute,
  errorElement: <RouteError />,
  hydrateFallbackElement: <RouteLoadingFallback />,
  loader: async ({ params }) => {
    const groupId = requireParam(params.groupId, paths.groups());
    const standardId = requireParam(params.standardId, paths.groupDetail(groupId));
    const imageId = requireParam(params.imageId, paths.standardDetail(groupId, standardId));

    await Promise.all([
      queryClient.ensureQueryData(getGroupQueryOptions(groupId)),
      queryClient.ensureQueryData(getStandardQueryOptions(standardId)),
      queryClient.ensureQueryData(getImageQueryOptions(imageId)),
    ]);

    return { groupId, standardId, imageId };
  },
};
