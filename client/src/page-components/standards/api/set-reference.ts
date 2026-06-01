import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { StandardImage } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export const setReference = (imageId: string): Promise<StandardImage> => {
  return client.patch(`/standards/images/${imageId}/reference`);
};

type Options = {
  groupId: string;
  standardId: string;
  mutationConfig?: MutationConfig<typeof setReference>;
};

export const useSetReference = ({ groupId, standardId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: setReference,
    onSuccess: async (...args) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.standards.detail(standardId) }),
      ]);
      notifySuccess("Изображение установлено как образец");
      await onSuccess?.(...args);
    },
    ...rest,
  });
};
