import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export const deleteModel = (modelId: string): Promise<void> => {
  return client.delete(`/yolo/${modelId}`);
};

type Options = {
  groupId: string;
  mutationConfig?: MutationConfig<typeof deleteModel>;
};

export const useDeleteModel = ({ groupId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: deleteModel,
    onSuccess: async (data, modelId, ctx, mutationCtx) => {
      qc.removeQueries({ queryKey: queryKeys.training.model(modelId) });

      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.training.models(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.tasks(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
      ]);

      notifySuccess("Модель удалена");
      onSuccess?.(data, modelId, ctx, mutationCtx);
    },
    ...rest,
  });
};
