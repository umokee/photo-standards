import { client } from "@/lib/api-client";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import { useMutation } from "@tanstack/react-query";
import { z } from "zod";

const exportModelSchema = z.object({
  modelId: z.string().trim().min(1, "Выберите модель"),
  fileName: z.string().trim().min(1, "Имя файла обязательно"),
});

export type ExportModelFormValues = z.input<typeof exportModelSchema>;
export type ExportModelPayload = z.output<typeof exportModelSchema>;

type BuildExportModelPayloadResult =
  | { ok: true; data: ExportModelPayload }
  | { ok: false; errors: Record<string, string> };

export const buildExportModelPayload = (
  values: ExportModelFormValues
): BuildExportModelPayloadResult => {
  const parsed = exportModelSchema.safeParse(values);

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

export const exportModel = async ({
  modelId,
  fileName,
}: ExportModelPayload): Promise<void> => {
  const blob = await client.get<Blob, Blob>(`/yolo/${modelId}/export`, {
    responseType: "blob",
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  try {
    link.href = url;
    link.download = fileName;
    document.body.append(link);
    link.click();
  } finally {
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }
};

type Options = {
  mutationConfig?: MutationConfig<typeof exportModel>;
};

export const useExportModel = ({ mutationConfig }: Options = {}) => {
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: exportModel,
    onSuccess: (data, vars, ctx, mutationCtx) => {
      notifySuccess(`Скачивание началось: ${vars.fileName}`);
      onSuccess?.(data, vars, ctx, mutationCtx);
    },
    ...rest,
  });
};
