import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { TrainingMetricsHistoryResponse } from "@/types/contracts";
import { queryOptions } from "@tanstack/react-query";

export const getModelMetricsHistory = (
  modelId: string
): Promise<TrainingMetricsHistoryResponse> => {
  return client.get(`/yolo/${modelId}/metrics-history`);
};

export const getModelMetricsHistoryQueryOptions = (modelId: string) =>
  queryOptions({
    queryKey: queryKeys.training.metricsHistory(modelId),
    queryFn: () => getModelMetricsHistory(modelId),
  });
