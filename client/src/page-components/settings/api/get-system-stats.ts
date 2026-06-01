import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { SystemStatsResponse } from "@/types/contracts/system";
import { queryOptions, useQuery } from "@tanstack/react-query";

export const getSystemStats = (): Promise<SystemStatsResponse> => {
  return client.get("/system/stats");
};

export const getSystemStatsQueryOptions = () =>
  queryOptions({
    queryKey: queryKeys.settings.system_stats(),
    queryFn: getSystemStats,
  });

export const useGetSystemStats = () => {
  return useQuery(getSystemStatsQueryOptions());
};
