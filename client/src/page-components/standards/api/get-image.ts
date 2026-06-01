import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { StandardImageDetail } from "@/types/contracts";
import { queryOptions, useSuspenseQuery } from "@tanstack/react-query";

export const getImage = (id: string): Promise<StandardImageDetail> => {
  return client.get(`/standards/images/${id}`);
};

export const getImageQueryOptions = (id: string) => {
  return queryOptions({
    queryKey: queryKeys.standards.image(id),
    queryFn: () => getImage(id),
  });
};

export const useGetImage = (id: string) => {
  return useSuspenseQuery(getImageQueryOptions(id));
};
