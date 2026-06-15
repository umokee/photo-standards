import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { GroupMutationResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";

const createGroupSchema = z.object({
  name: z.string().trim().min(1, "Укажите название"),
  description: z.string().transform((value) => {
    const next = value.trim();
    return next === "" ? null : next;
  }),
});

export type CreateGroupFormValues = z.input<typeof createGroupSchema>;
export type CreateGroupPayload = z.output<typeof createGroupSchema>;

type BuildCreateGroupPayloadResult =
  | { ok: true; data: CreateGroupPayload }
  | { ok: false; errors: Record<string, string> };

export const buildCreateGroupPayload = (
  values: CreateGroupFormValues
): BuildCreateGroupPayloadResult => {
  const parsed = createGroupSchema.safeParse(values);

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

export const createGroup = (data: CreateGroupPayload): Promise<GroupMutationResponse> => {
  return client.post("/groups", data);
};

type Options = {
  mutationConfig?: MutationConfig<typeof createGroup>;
};

export const useCreateGroup = ({ mutationConfig }: Options = {}) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: createGroup,
    onSuccess: async (...args) => {
      await qc.invalidateQueries({ queryKey: queryKeys.groups.all() });
      notifySuccess("Группа успешно создана");
      await onSuccess?.(...args);
    },
    ...rest,
  });
};
