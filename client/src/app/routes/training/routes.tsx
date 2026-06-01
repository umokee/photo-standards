import { queryClient } from "@/lib/query-client";
import { getGroupQueryOptions } from "@/page-components/groups/api/get-group";
import { getGroupsQueryOptions } from "@/page-components/groups/api/get-groups";
import { getModelQueryOptions } from "@/page-components/models/api/get-ml";
import { getModelsQueryOptions } from "@/page-components/models/api/get-models";
import { getTasksQueryOptions } from "@/page-components/tasks/api/get-tasks";
import { RouteObject } from "react-router-dom";
import { paths } from "../../paths";
import { requireParam } from "../../route-params";
import RouteError from "../route-error";
import RouteLoadingFallback from "../route-loading-fallback";

const TrainingLayoutRoute = () => import("./_training-layout");
const TrainingIndexRoute = () => import("./_training-index");
const TrainingDetailRoute = () => import("./_training-detail");
const ModelDetailRoute = () => import("./_model-detail");

export const trainingRoute: RouteObject = {
  path: paths.training(),
  lazy: TrainingLayoutRoute,
  errorElement: <RouteError />,
  hydrateFallbackElement: <RouteLoadingFallback />,
  loader: async () => {
    await queryClient.ensureQueryData(getGroupsQueryOptions());
  },
  children: [
    {
      index: true,
      lazy: TrainingIndexRoute,
    },
    {
      path: ":groupId",
      lazy: TrainingDetailRoute,
      loader: async ({ params }) => {
        const groupId = requireParam(params.groupId, paths.training());

        await Promise.all([
          queryClient.ensureQueryData(getGroupQueryOptions(groupId)),
          queryClient.ensureQueryData(getModelsQueryOptions(groupId)),
          queryClient.ensureQueryData(getTasksQueryOptions(groupId)),
        ]);

        return { groupId };
      },
      children: [
        {
          index: true,
          lazy: ModelDetailRoute,
          loader: async () => {
            return { modelId: null };
          },
        },
        {
          path: "models/:modelId",
          lazy: ModelDetailRoute,
          loader: async ({ params }) => {
            const groupId = requireParam(params.groupId, paths.training());
            const modelId = requireParam(params.modelId, paths.trainingGroup(groupId));

            await queryClient.ensureQueryData(getModelQueryOptions(modelId));

            return { modelId };
          },
        },
      ],
    },
  ],
};
