import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { TaskResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";

type TaskActionResponse = { task: TaskResponse };

export const resumeTask = (taskId: string): Promise<TaskActionResponse> => {
  return client.post(`/tasks/${taskId}/resume`);
};

type Options = {
  groupId: string;
  mutationConfig?: MutationConfig<typeof resumeTask>;
};

export const useResumeTask = ({ groupId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: resumeTask,
    async onSuccess(data, taskId, onMutateResult, context) {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.training.task(taskId) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.tasks(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.models(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
      ]);

      notifySuccess("Обучение возобновлено");
      await onSuccess?.(data, taskId, onMutateResult, context);
    },
    ...rest,
  });
};
