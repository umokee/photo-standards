import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export const deleteStandard = (id: string): Promise<void> => {
  return client.delete(`/standards/${id}`);
};

type Options = {
  groupId: string;
  mutationConfig?: MutationConfig<typeof deleteStandard>;
};

export const useDeleteStandard = ({ groupId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: deleteStandard,
    onSuccess: async (data, vars, ctx, mutation) => {
      qc.removeQueries({ queryKey: queryKeys.standards.detail(vars) });

      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.groups.all() }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
      ]);

      notifySuccess("Эталон успешно удален");
      await onSuccess?.(data, vars, ctx, mutation);
    },
    ...rest,
  });
};
