import { getConstantsOrThrow } from "@/constants";
import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { Angle, StandardMutationResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";

const updateStandardSchema = z.object({
  name: z.string().max(255, "Название слишком длинное"),
  angle: z.string().nullable(),
});

export type UpdateStandardFormValues = {
  name: string;
  angle: Angle | null;
};

export type UpdateStandardInput = {
  id: string;
  data: {
    name?: string;
    angle?: Angle | null;
  };
};

type BuildUpdateStandardPayloadResult =
  | { ok: true; data: UpdateStandardInput["data"] | null }
  | { ok: false; errors: Record<string, string> };

export const buildUpdateStandardPayload = (
  current: UpdateStandardFormValues,
  initial: UpdateStandardFormValues
): BuildUpdateStandardPayloadResult => {
  const currentParsed = updateStandardSchema
    .superRefine((input, ctx) => {
      if (!input.name.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["name"],
          message: "Укажите название",
        });
      }

      if (!isAllowedAngle(input.angle)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["angle"],
          message: "Выберите ракурс из списка",
        });
      }
    })
    .transform((input) => ({
      name: input.name.trim(),
      angle: normalizeAngle(input.angle) ?? null,
    }))
    .safeParse(current);

  if (!currentParsed.success) {
    return {
      ok: false,
      errors: Object.fromEntries(
        currentParsed.error.issues.map((issue) => [String(issue.path[0] ?? "form"), issue.message])
      ),
    };
  }

  const initialParsed = updateStandardSchema
    .transform((input) => ({
      name: input.name.trim(),
      angle: normalizeAngle(input.angle) ?? null,
    }))
    .parse(initial);

  const changed: UpdateStandardInput["data"] = {};

  if (!Object.is(currentParsed.data.name, initialParsed.name)) {
    changed.name = currentParsed.data.name;
  }

  if (!Object.is(currentParsed.data.angle, initialParsed.angle)) {
    changed.angle = currentParsed.data.angle;
  }

  return {
    ok: true,
    data: Object.keys(changed).length > 0 ? changed : null,
  };
};

export const updateStandard = ({
  id,
  data,
}: UpdateStandardInput): Promise<StandardMutationResponse> => {
  return client.put(`/standards/${id}`, data);
};

type Options = {
  groupId: string;
  mutationConfig?: MutationConfig<typeof updateStandard>;
};

export const useUpdateStandard = ({ groupId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: updateStandard,
    onSuccess: async (data, vars, ctx, mutation) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.standards.detail(vars.id) }),
      ]);
      notifySuccess("Эталон успешно обновлен");
      await onSuccess?.(data, vars, ctx, mutation);
    },
    ...rest,
  });
};

const normalizeAngle = (angle: string | null): Angle | undefined => {
  const next = angle?.trim() ?? "";
  return next === "" ? undefined : next;
};

const isAllowedAngle = (angle: string | null): boolean => {
  const next = angle?.trim() ?? "";
  if (next === "") {
    return true;
  }

  return getConstantsOrThrow().standards.angles.values.includes(next);
};
