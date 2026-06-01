import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { MlModel } from "@/types/contracts";
import { queryOptions, useSuspenseQuery } from "@tanstack/react-query";

export const getModels = (groupId: string): Promise<MlModel[]> => {
  return client.get("/yolo", {
    params: { group_id: groupId },
  });
};

export const getModelsQueryOptions = (groupId: string) => {
  return queryOptions({
    queryKey: queryKeys.training.models(groupId),
    queryFn: () => getModels(groupId),
  });
};

export const useGetModels = (groupId: string) => {
  return useSuspenseQuery(getModelsQueryOptions(groupId));
};
