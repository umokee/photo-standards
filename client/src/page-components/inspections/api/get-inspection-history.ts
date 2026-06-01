import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { InspectionHistoryItem } from "@/types/contracts";
import { queryOptions, useSuspenseQuery } from "@tanstack/react-query";

export const getInspectionHistory = (
  groupId: string | null = null
): Promise<InspectionHistoryItem[]> => {
  return client.get("/yolo/inspection/history", {
    params: groupId ? { group_id: groupId } : undefined,
  }) as Promise<InspectionHistoryItem[]>;
};

export const getInspectionHistoryQueryOptions = (groupId: string | null = null) =>
  queryOptions({
    queryKey: queryKeys.inspections.history(groupId),
    queryFn: () => getInspectionHistory(groupId),
  });

export const useGetInspectionHistory = (groupId: string | null = null) => {
  return useSuspenseQuery(getInspectionHistoryQueryOptions(groupId));
};
