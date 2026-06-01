import { queryClient } from "@/lib/query-client";
import { getGroupQueryOptions } from "@/page-components/groups/api/get-group";
import { getGroupsQueryOptions } from "@/page-components/groups/api/get-groups";
import { getStandardQueryOptions } from "@/page-components/standards/api/get-standard";
import { RouteObject } from "react-router-dom";
import { paths } from "../../paths";
import { requireParam } from "../../route-params";
import RouteError from "../route-error";
import RouteLoadingFallback from "../route-loading-fallback";

const GroupsLayoutRoute = () => import("./_groups-layout");
const GroupsIndexRoute = () => import("./_groups-index");
const GroupDetailRoute = () => import("./_group-detail");
const StandardDetailRoute = () => import("./_standard-detail");

export const groupsRoute: RouteObject = {
  path: paths.groups(),
  lazy: GroupsLayoutRoute,
  errorElement: <RouteError />,
  hydrateFallbackElement: <RouteLoadingFallback />,
  loader: async () => {
    await queryClient.ensureQueryData(getGroupsQueryOptions());
  },
  children: [
    {
      index: true,
      lazy: GroupsIndexRoute,
    },
    {
      path: ":groupId",
      lazy: GroupDetailRoute,
      loader: async ({ params }) => {
        const groupId = requireParam(params.groupId, paths.groups());

        await queryClient.ensureQueryData(getGroupQueryOptions(groupId));

        return { groupId };
      },
      children: [
        {
          index: true,
          lazy: StandardDetailRoute,
          loader: async () => {
            return { standardId: null };
          },
        },
        {
          path: "standards/:standardId",
          lazy: StandardDetailRoute,
          loader: async ({ params }) => {
            const groupId = requireParam(params.groupId, paths.groups());
            const standardId = requireParam(params.standardId, paths.groupDetail(groupId));

            await Promise.all([
              queryClient.ensureQueryData(getGroupQueryOptions(groupId)),
              queryClient.ensureQueryData(getStandardQueryOptions(standardId)),
            ]);

            return { standardId };
          },
        },
      ],
    },
  ],
};
