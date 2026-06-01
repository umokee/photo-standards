import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { MutationConfig } from "@/lib/react-query";
import { SegmentClassWithPoints, StandardImageDetail } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export type AnnotateSegmentClassInput = {
  segmentClassId: string;
  imageId: string;
  points: number[][][];
};

export const annotateSegmentClassMutationKey = ["segment-class-annotation"] as const;

export const annotateSegmentClass = ({
  segmentClassId,
  imageId,
  points,
}: AnnotateSegmentClassInput): Promise<SegmentClassWithPoints> => {
  return client.put(`/segment-classes/${segmentClassId}/annotations/${imageId}`, {
    points,
  });
};

type Options = {
  groupId: string;
  standardId: string;
  mutationConfig?: MutationConfig<typeof annotateSegmentClass>;
};

export const useAnnotateSegmentClass = ({ groupId, standardId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationKey: annotateSegmentClassMutationKey,
    mutationFn: annotateSegmentClass,
    onSuccess: async (data, vars, ctx, mutation) => {
      qc.setQueryData<StandardImageDetail | undefined>(
        queryKeys.standards.image(vars.imageId),
        (current) =>
          current
            ? {
                ...current,
                segment_classes: current.segment_classes.map((item) =>
                  item.id === data.id ? data : item
                ),
              }
            : current
      );

      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.standards.detail(standardId) }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.all() }),
      ]);

      await onSuccess?.(data, vars, ctx, mutation);
    },
    ...rest,
  });
};
