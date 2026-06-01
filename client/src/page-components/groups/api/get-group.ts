import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { GroupDetail } from "@/types/contracts";
import { queryOptions, useSuspenseQuery } from "@tanstack/react-query";

export const getGroup = (id: string): Promise<GroupDetail> => {
  return client.get(`/groups/${id}`);
};

export const getGroupQueryOptions = (id: string) => {
  return queryOptions({
    queryKey: queryKeys.groups.detail(id),
    queryFn: () => getGroup(id),
  });
};

export const useGetGroup = (id: string) => {
  return useSuspenseQuery(getGroupQueryOptions(id));
};
