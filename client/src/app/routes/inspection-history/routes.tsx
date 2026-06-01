import { queryClient } from "@/lib/query-client";
import { getGroupsQueryOptions } from "@/page-components/groups/api/get-groups";
import { getInspectionQueryOptions } from "@/page-components/inspections/api/get-inspection";
import { getInspectionHistoryQueryOptions } from "@/page-components/inspections/api/get-inspection-history";
import { RouteObject } from "react-router-dom";
import { paths } from "../../paths";
import { requireParam } from "../../route-params";
import RouteError from "../route-error";
import RouteLoadingFallback from "../route-loading-fallback";

const InspectionHistoryLayoutRoute = () => import("./_inspection-history-layout");
const InspectionHistoryIndexRoute = () => import("./_inspection-history-index");
const InspectionHistoryGroupRoute = () => import("./_inspection-history-group");
const InspectionHistoryDetailRoute = () => import("./_inspection-history-detail");

export const inspectionHistoryRoute: RouteObject = {
  path: paths.inspectionHistory(),
  lazy: InspectionHistoryLayoutRoute,
  errorElement: <RouteError />,
  hydrateFallbackElement: <RouteLoadingFallback />,
  loader: async () => {
    await queryClient.ensureQueryData(getGroupsQueryOptions());
  },
  children: [
    {
      index: true,
      lazy: InspectionHistoryIndexRoute,
    },
    {
      path: ":groupId",
      lazy: InspectionHistoryGroupRoute,
      loader: async ({ params }) => {
        const groupId = requireParam(params.groupId, paths.inspectionHistory());

        await queryClient.ensureQueryData(getInspectionHistoryQueryOptions(groupId));

        return { groupId };
      },
      children: [
        {
          index: true,
          lazy: InspectionHistoryDetailRoute,
          loader: async () => {
            return { inspectionId: null };
          },
        },
        {
          path: ":inspectionId",
          lazy: InspectionHistoryDetailRoute,
          loader: async ({ params }) => {
            const groupId = requireParam(params.groupId, paths.inspectionHistory());
            const inspectionId = requireParam(
              params.inspectionId,
              paths.inspectionHistoryGroup(groupId)
            );

            await Promise.all([
              queryClient.ensureQueryData(getInspectionHistoryQueryOptions(groupId)),
              queryClient.ensureQueryData(getInspectionQueryOptions(inspectionId)),
            ]);

            return { inspectionId };
          },
        },
      ],
    },
  ],
};
