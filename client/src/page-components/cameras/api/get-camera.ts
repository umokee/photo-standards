import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { Camera } from "@/types/contracts";
import { queryOptions, useSuspenseQuery } from "@tanstack/react-query";

export const getCamera = (id: string): Promise<Camera> => {
  return client.get(`/cameras/${id}`);
};

export const getCameraQueryOptions = (id: string) => {
  return queryOptions({
    queryKey: queryKeys.cameras.detail(id),
    queryFn: () => getCamera(id),
    staleTime: 30_000,
  });
};

export const useGetCamera = (id: string) => {
  return useSuspenseQuery(getCameraQueryOptions(id));
};
