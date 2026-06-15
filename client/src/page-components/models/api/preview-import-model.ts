import { client } from "@/lib/api-client";
import type { MutationConfig } from "@/lib/react-query";
import type { ModelImportPreviewResponse } from "@/types/contracts";
import { useMutation } from "@tanstack/react-query";
import { z } from "zod";

const previewImportModelSchema = z
  .object({
    group_id: z.string().trim().min(1, "Выберите группу"),
    weights: z.instanceof(File).nullable(),
  })
  .superRefine((values, ctx) => {
    if (!values.weights) {
      ctx.addIssue({
        code: "custom",
        path: ["weights"],
        message: "Выберите файл .pt",
      });
      return;
    }

    if (!values.weights.name.toLowerCase().endsWith(".pt")) {
      ctx.addIssue({
        code: "custom",
        path: ["weights"],
        message: "Выберите файл в формате .pt",
      });
    }
  });

export type PreviewImportModelFormValues = z.input<typeof previewImportModelSchema>;
export type PreviewImportModelPayload = Omit<
  z.output<typeof previewImportModelSchema>,
  "weights"
> & {
  weights: File;
};

type BuildPreviewImportModelPayloadResult =
  | { ok: true; data: PreviewImportModelPayload }
  | { ok: false; errors: Record<string, string> };

export const buildPreviewImportModelPayload = (
  values: PreviewImportModelFormValues
): BuildPreviewImportModelPayloadResult => {
  const parsed = previewImportModelSchema.safeParse(values);

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
    data: {
      ...parsed.data,
      weights: parsed.data.weights as File,
    },
  };
};

export const previewImportModel = async (
  input: PreviewImportModelPayload
): Promise<ModelImportPreviewResponse> => {
  const formData = new FormData();

  formData.append("group_id", input.group_id);
  formData.append("weights", input.weights);

  return client.post("/yolo/import/preview", formData);
};

type Options = {
  mutationConfig?: MutationConfig<typeof previewImportModel>;
};

export const usePreviewImportModel = ({ mutationConfig }: Options = {}) => {
  return useMutation({
    mutationFn: previewImportModel,
    ...mutationConfig,
  });
};
