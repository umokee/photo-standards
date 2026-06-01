import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { MlModel } from "@/types/contracts";
import { queryOptions, useQuery } from "@tanstack/react-query";

export const getModel = (modelId: string): Promise<MlModel> => {
  return client.get(`/yolo/${modelId}`);
};

export const getModelQueryOptions = (modelId: string) => {
  return queryOptions({
    queryKey: queryKeys.training.model(modelId),
    queryFn: () => getModel(modelId),
  });
};

export const useGetModel = (modelId: string | null) => {
  return useQuery({
    queryKey: queryKeys.training.model(modelId ?? ""),
    queryFn: () => getModel(modelId),
    enabled: !!modelId,
  });
};
