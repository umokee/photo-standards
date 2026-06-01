import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import { TaskResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";

type TaskActionResponse = { task: TaskResponse };

export const cancelTask = (taskId: string): Promise<TaskActionResponse> => {
  return client.post(`/tasks/${taskId}/cancel`);
};

type Options = {
  groupId: string;
  mutationConfig?: MutationConfig<typeof cancelTask>;
};

export const useCancelTask = ({ groupId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: cancelTask,
    onSuccess: (data, taskId, onMutateResult, context) => {
      qc.invalidateQueries({ queryKey: queryKeys.training.task(taskId) });
      qc.invalidateQueries({ queryKey: queryKeys.training.tasks(groupId) });
      qc.invalidateQueries({ queryKey: queryKeys.training.models(groupId) });
      qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) });
      notifySuccess("Задача отменена");
      onSuccess?.(data, taskId, onMutateResult, context);
    },
    ...rest,
  });
};
