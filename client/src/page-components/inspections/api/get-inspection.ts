import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { InspectionResult } from "@/types/contracts";
import { queryOptions, useSuspenseQuery } from "@tanstack/react-query";

export const getInspection = (inspectionId: string): Promise<InspectionResult> => {
  return client.get(`/yolo/inspection/${inspectionId}`) as Promise<InspectionResult>;
};

export const getInspectionQueryOptions = (inspectionId: string) =>
  queryOptions({
    queryKey: queryKeys.inspections.detail(inspectionId),
    queryFn: () => getInspection(inspectionId),
  });

export const useGetInspection = (inspectionId: string) => {
  return useSuspenseQuery(getInspectionQueryOptions(inspectionId));
};
