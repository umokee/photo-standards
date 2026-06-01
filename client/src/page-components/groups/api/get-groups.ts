import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { GroupListItem } from "@/types/contracts";
import { queryOptions, useSuspenseQuery } from "@tanstack/react-query";

export const getGroups = (): Promise<GroupListItem[]> => {
  return client.get("/groups");
};

export const getGroupsQueryOptions = () => {
  return queryOptions({
    queryKey: queryKeys.groups.all(),
    queryFn: getGroups,
  });
};

export const useGetGroups = () => {
  return useSuspenseQuery(getGroupsQueryOptions());
};
