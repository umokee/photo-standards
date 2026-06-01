import { queryClient } from "@/lib/query-client";
import { getGroupQueryOptions } from "@/page-components/groups/api/get-group";
import { getGroupsQueryOptions } from "@/page-components/groups/api/get-groups";
import { getStandardQueryOptions } from "@/page-components/standards/api/get-standard";
import { RouteObject, redirect } from "react-router-dom";
import { paths } from "../../paths";
import { requireInspectionMode, requireParam } from "../../route-params";
import RouteError from "../route-error";
import RouteLoadingFallback from "../route-loading-fallback";

const InspectionLayoutRoute = () => import("./_inspection-layout");
const InspectionIndexRoute = () => import("./_inspection-index");
const InspectionGroupRoute = () => import("./_inspection-group");
const InspectionStandardRoute = () => import("./_inspection-standard");

export const inspectionRedirectRoute: RouteObject = {
  path: paths.inspection(),
  loader: async () => {
    throw redirect(paths.inspectionMode("photo"));
  },
};

export const inspectionModeRoute: RouteObject = {
  path: `${paths.inspection()}/:mode`,
  lazy: InspectionLayoutRoute,
  errorElement: <RouteError />,
  hydrateFallbackElement: <RouteLoadingFallback />,
  loader: async ({ params }) => {
    const mode = requireInspectionMode(params.mode);

    await queryClient.ensureQueryData(getGroupsQueryOptions());

    return { mode };
  },
  children: [
    {
      index: true,
      lazy: InspectionIndexRoute,
    },
    {
      path: "groups/:groupId",
      lazy: InspectionGroupRoute,
      loader: async ({ params }) => {
        const mode = requireInspectionMode(params.mode);
        const groupId = requireParam(params.groupId, paths.inspectionMode(mode));

        await queryClient.ensureQueryData(getGroupQueryOptions(groupId));

        return { mode, groupId };
      },
    },
    {
      path: "groups/:groupId/standards/:standardId",
      lazy: InspectionStandardRoute,
      loader: async ({ params }) => {
        const mode = requireInspectionMode(params.mode);
        const groupId = requireParam(params.groupId, paths.inspectionMode(mode));
        const standardId = requireParam(params.standardId, paths.inspectionGroup(mode, groupId));

        await Promise.all([
          queryClient.ensureQueryData(getGroupQueryOptions(groupId)),
          queryClient.ensureQueryData(getStandardQueryOptions(standardId)),
        ]);

        return { mode, groupId, standardId };
      },
    },
  ],
};
