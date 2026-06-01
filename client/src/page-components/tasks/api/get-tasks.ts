import { useEffect, useRef } from "react";

import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { TaskResponse } from "@/types/contracts";
import { queryOptions, useQuery, useQueryClient } from "@tanstack/react-query";

import { isActiveTaskStatus, isTerminalTaskStatus } from "../lib/task-helpers";

type GetTasksParams = {
  groupId: string;
  type?: string | null;
};

export const getTasks = ({ groupId, type }: GetTasksParams): Promise<TaskResponse[]> => {
  return client.get("/tasks", {
    params: {
      group_id: groupId,
      ...(type ? { type } : {}),
    },
  });
};

export const getTasksQueryOptions = (groupId: string) =>
  queryOptions({
    queryKey: queryKeys.training.tasks(groupId),
    queryFn: () => getTasks({ groupId }),
  });

export const useGetTasks = (groupId: string) => {
  const queryClient = useQueryClient();
  const previousTaskStatusesRef = useRef(new Map<string, string>());
  const query = useQuery(getTasksQueryOptions(groupId));

  useEffect(() => {
    const nextTasks = query.data ?? [];
    const completedModelIds = new Set<string>();

    for (const task of nextTasks) {
      const previousStatus = previousTaskStatusesRef.current.get(task.id);
      if (!isActiveTaskStatus(previousStatus) || !isTerminalTaskStatus(task.status)) {
        continue;
      }
      if (task.entity_type === "ml_model" && task.entity_id) {
        completedModelIds.add(task.entity_id);
      }
    }

    previousTaskStatusesRef.current = new Map(nextTasks.map((task) => [task.id, task.status]));

    if (completedModelIds.size === 0) {
      return;
    }

    void Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.training.models(groupId) }),
      queryClient.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
      ...Array.from(completedModelIds).map((modelId) =>
        queryClient.invalidateQueries({ queryKey: queryKeys.training.model(modelId) })
      ),
    ]);
  }, [groupId, query.data, queryClient]);

  return query;
};
