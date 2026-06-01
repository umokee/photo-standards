import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { GroupMutationResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";

const updateGroupSchema = z.object({
  name: z.string().trim().min(1, "Название обязательно"),
  description: z.string().transform((value) => {
    const next = value.trim();
    return next === "" ? null : next;
  }),
});

export type UpdateGroupFormValues = z.input<typeof updateGroupSchema>;
export type UpdateGroupPayload = Partial<z.output<typeof updateGroupSchema>>;

type BuildUpdateGroupPayloadResult =
  | { ok: true; data: UpdateGroupPayload | null }
  | { ok: false; errors: Record<string, string> };

export const buildUpdateGroupPayload = (
  current: UpdateGroupFormValues,
  initial: UpdateGroupFormValues
): BuildUpdateGroupPayloadResult => {
  const currentParsed = updateGroupSchema.safeParse(current);
  if (!currentParsed.success) {
    return {
      ok: false,
      errors: Object.fromEntries(
        currentParsed.error.issues.map((issue) => [String(issue.path[0] ?? "form"), issue.message])
      ),
    };
  }

  const initialParsed = updateGroupSchema.parse(initial);
  const changed = buildChangedGroupData(currentParsed.data, initialParsed);

  return {
    ok: true,
    data: changed,
  };
};

export const updateGroup = ({
  id,
  data,
}: {
  id: string;
  data: UpdateGroupPayload;
}): Promise<GroupMutationResponse> => {
  return client.put(`/groups/${id}`, data);
};

type Options = {
  mutationConfig?: MutationConfig<typeof updateGroup>;
};

export const useUpdateGroup = ({ mutationConfig }: Options = {}) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: updateGroup,
    onSuccess: async (data, vars, ctx, mutation) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.groups.all() }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(vars.id) }),
      ]);
      notifySuccess("Группа успешно обновлена");
      await onSuccess?.(data, vars, ctx, mutation);
    },
    ...rest,
  });
};

const buildChangedGroupData = (
  current: z.output<typeof updateGroupSchema>,
  initial: z.output<typeof updateGroupSchema>
): UpdateGroupPayload | null => {
  const changed: UpdateGroupPayload = {};

  for (const key of Object.keys(current) as (keyof typeof current)[]) {
    if (!Object.is(current[key], initial[key])) {
      changed[key] = current[key];
    }
  }

  return Object.keys(changed).length > 0 ? changed : null;
};
