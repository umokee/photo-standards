import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export const deleteGroup = (id: string): Promise<void> => {
  return client.delete(`/groups/${id}`);
};

type Options = {
  mutationConfig?: MutationConfig<typeof deleteGroup>;
};

export const useDeleteGroup = ({ mutationConfig }: Options = {}) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: deleteGroup,
    onSuccess: async (data, groupId, ctx, mutation) => {
      qc.removeQueries({ queryKey: queryKeys.groups.detail(groupId) });
      await qc.invalidateQueries({ queryKey: queryKeys.groups.all() });
      notifySuccess("Группа успешно удалена");
      await onSuccess?.(data, groupId, ctx, mutation);
    },
    ...rest,
  });
};
