import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { Camera } from "@/types/contracts";
import { queryOptions, useSuspenseQuery } from "@tanstack/react-query";

export const getCameras = (): Promise<Camera[]> => {
  return client.get("/cameras");
};

export const getCamerasQueryOptions = () => {
  return queryOptions({
    queryKey: queryKeys.cameras.all(),
    queryFn: getCameras,
    staleTime: 30_000,
  });
};

export const useGetCameras = () => {
  return useSuspenseQuery(getCamerasQueryOptions());
};
