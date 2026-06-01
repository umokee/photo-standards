import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { TaskResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";

type TaskActionResponse = { task: TaskResponse };

export const pauseTask = (taskId: string): Promise<TaskActionResponse> => {
  return client.post(`/tasks/${taskId}/pause`);
};

type Options = {
  groupId: string;
  mutationConfig?: MutationConfig<typeof pauseTask>;
};

export const usePauseTask = ({ groupId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: pauseTask,
    async onSuccess(data, taskId, onMutateResult, context) {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.training.task(taskId) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.tasks(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.models(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
      ]);

      notifySuccess("Обучение приостановлено");
      await onSuccess?.(data, taskId, onMutateResult, context);
    },
    ...rest,
  });
};
