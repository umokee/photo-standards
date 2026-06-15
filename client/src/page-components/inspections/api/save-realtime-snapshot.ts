import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { InspectionSaveResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";

const saveRealtimeSnapshotSchema = z.object({
  session_id: z.string().trim().min(1, "Не удалось определить сессию реального времени"),
  notes: z.string().nullable().optional().transform((value) => value?.trim() || null),
});

export type SaveRealtimeSnapshotFormValues = z.input<typeof saveRealtimeSnapshotSchema>;
export type SaveRealtimeSnapshotInput = z.output<typeof saveRealtimeSnapshotSchema>;

type BuildSaveRealtimeSnapshotPayloadResult =
  | { ok: true; data: SaveRealtimeSnapshotInput }
  | { ok: false; errors: Record<string, string> };

export const buildSaveRealtimeSnapshotPayload = (
  values: SaveRealtimeSnapshotFormValues
): BuildSaveRealtimeSnapshotPayloadResult => {
  const parsed = saveRealtimeSnapshotSchema.safeParse(values);

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

export const saveRealtimeSnapshot = async (
  input: SaveRealtimeSnapshotInput
): Promise<InspectionSaveResponse> => {
  return client.post(`/yolo/inspection/realtime/sessions/${input.session_id}/snapshot`, {
    notes: input.notes,
  });
};

type Options = {
  mutationConfig?: MutationConfig<typeof saveRealtimeSnapshot>;
};

export const useSaveRealtimeSnapshot = ({ mutationConfig }: Options = {}) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: saveRealtimeSnapshot,
    onSuccess: async (...args) => {
      await Promise.all([
        qc.invalidateQueries({
          queryKey: queryKeys.inspections.historyRoot(),
        }),
        qc.invalidateQueries({
          queryKey: queryKeys.groups.all(),
        }),
      ]);
      notifySuccess("Текущий кадр сохранён");
      onSuccess?.(...args);
    },
    ...rest,
  });
};
