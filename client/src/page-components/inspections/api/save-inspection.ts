import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { InspectionSaveResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";

const saveInspectionSchema = z.object({
  task_id: z.string().trim().min(1, "Не удалось определить задачу проверки"),
  notes: z.string().nullable().optional().transform((value) => value?.trim() || null),
});

export type SaveInspectionFormValues = z.input<typeof saveInspectionSchema>;
export type SaveInspectionInput = z.output<typeof saveInspectionSchema>;

export const saveInspection = async (
  input: SaveInspectionInput
): Promise<InspectionSaveResponse> => {
  return client.post("/yolo/inspection/save", input);
};

type BuildSaveInspectionPayloadResult =
  | { ok: true; data: SaveInspectionInput }
  | { ok: false; errors: Record<string, string> };

export const buildSaveInspectionPayload = (
  values: SaveInspectionFormValues
): BuildSaveInspectionPayloadResult => {
  const parsed = saveInspectionSchema.safeParse(values);

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

type Options = {
  mutationConfig?: MutationConfig<typeof saveInspection>;
};

export const useSaveInspection = ({ mutationConfig }: Options = {}) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: saveInspection,
    onSuccess: async (...args) => {
      await Promise.all([
        qc.invalidateQueries({
          queryKey: queryKeys.inspections.historyRoot(),
        }),
        qc.invalidateQueries({
          queryKey: queryKeys.groups.all(),
        }),
      ]);
      notifySuccess("Результат проверки сохранён");
      onSuccess?.(...args);
    },
    ...rest,
  });
};
