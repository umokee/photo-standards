import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import { MlModel } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export const activateModel = (modelId: string): Promise<MlModel> => {
  return client.put(`/yolo/${modelId}/activate`);
};

type Options = {
  groupId: string;
  mutationConfig?: MutationConfig<typeof activateModel>;
};

export const useActivateModel = ({ groupId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: activateModel,
    onSuccess: async (data, vars, ctx, mutationCtx) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.training.models(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.model(data.id) }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
      ]);

      notifySuccess("Модель активирована");
      onSuccess?.(data, vars, ctx, mutationCtx);
    },
    ...rest,
  });
};
