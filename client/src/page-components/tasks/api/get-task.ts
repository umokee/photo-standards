import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { TaskResponse } from "@/types/contracts";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { isActiveTaskStatus } from "../lib/task-helpers";

export const getTask = (taskId: string): Promise<TaskResponse> => {
  return client.get(`/tasks/${taskId}`);
};

export const getTaskQueryOptions = (taskId: string) =>
  queryOptions({
    queryKey: queryKeys.training.task(taskId),
    queryFn: () => getTask(taskId),
    enabled: !!taskId,
  });

export const useGetTask = (taskId: string | null) => {
  return useQuery({
    ...getTaskQueryOptions(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      const task = query.state.data as TaskResponse | undefined;
      return isActiveTaskStatus(task?.status) ? 1000 : false;
    },
  });
};
