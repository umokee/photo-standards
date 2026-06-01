import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { MutationConfig, notifySuccess } from "@/lib/react-query";
import { SaveSegmentClassesResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export type SaveSegmentClassesInput = {
  groupId: string;
  categories: {
    id?: string | null;
    name: string;
    segment_classes: {
      id?: string | null;
      name: string;
      hue: number;
    }[];
  }[];
  ungroupedClasses: {
    id?: string | null;
    name: string;
    hue: number;
  }[];
  deletedCategoryIds: string[];
  deletedClassIds: string[];
};

export const saveSegmentClasses = ({
  groupId,
  categories,
  ungroupedClasses,
  deletedCategoryIds,
  deletedClassIds,
}: SaveSegmentClassesInput): Promise<SaveSegmentClassesResponse> =>
  client.put(`/segment-classes/group/${groupId}`, {
    categories,
    ungrouped_classes: ungroupedClasses,
    deleted_category_ids: deletedCategoryIds,
    deleted_class_ids: deletedClassIds,
  });

type Options = {
  groupId: string;
  standardId?: string;
  imageId?: string;
  mutationConfig?: MutationConfig<typeof saveSegmentClasses>;
};

export const useSaveSegmentClasses = ({
  groupId,
  standardId,
  imageId,
  mutationConfig,
}: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: saveSegmentClasses,
    onSuccess: (data, vars, ctx, mutation) => {
      qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) });
      qc.invalidateQueries({ queryKey: queryKeys.groups.all() });

      if (standardId) {
        qc.invalidateQueries({ queryKey: queryKeys.standards.detail(standardId) });
      }

      if (imageId) {
        qc.invalidateQueries({ queryKey: queryKeys.standards.image(imageId) });
      }

      notifySuccess("Классы успешно сохранены");
      onSuccess?.(data, vars, ctx, mutation);
    },
    ...rest,
  });
};
