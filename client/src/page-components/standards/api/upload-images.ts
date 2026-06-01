import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { StandardImage } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";

const uploadImagesSchema = z.object({
  standardId: z.string().trim().min(1, "Не удалось определить эталон"),
  images: z.array(z.instanceof(File)).min(1, "Добавьте хотя бы одно изображение"),
});

export type UploadImagesInput = {
  standardId: string;
  images: File[];
};

export type UploadImagesFormValues = {
  standardId: string;
  images: File[] | null;
};

type BuildUploadImagesPayloadResult =
  | { ok: true; data: UploadImagesInput }
  | { ok: false; errors: Record<string, string> };

export const buildUploadImagesPayload = (
  values: UploadImagesFormValues
): BuildUploadImagesPayloadResult => {
  const parsed = uploadImagesSchema.safeParse({
    standardId: values.standardId,
    images: values.images ?? [],
  });

  if (!parsed.success) {
    return {
      ok: false,
      errors: Object.fromEntries(
        parsed.error.issues.map((issue) => [String(issue.path[0] ?? "form"), issue.message])
      ),
    };
  }

  return {
    ok: true,
    data: parsed.data,
  };
};

export const uploadImages = ({
  standardId,
  images,
}: UploadImagesInput): Promise<StandardImage[]> => {
  const form = new FormData();
  images.forEach((image) => form.append("images", image));
  return client.post(`/standards/${standardId}/images`, form);
};

type Options = {
  groupId: string;
  mutationConfig?: MutationConfig<typeof uploadImages>;
};

export const useUploadImages = ({ groupId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: uploadImages,
    onSuccess: async (data, vars, ctx, mutation) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.groups.all() }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.standards.detail(vars.standardId) }),
      ]);
      notifySuccess("Изображения успешно загружены");
      await onSuccess?.(data, vars, ctx, mutation);
    },
    ...rest,
  });
};
