import { getConstantsOrThrow } from "@/constants";
import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { Angle, StandardMutationResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";

const createStandardSchema = z.object({
  groupId: z.string(),
  name: z.string(),
  angle: z.string().nullable(),
});

export type CreateStandardFormValues = {
  groupId: string;
  name: string;
  angle: Angle | null;
};

export type CreateStandardInput = {
  groupId: string;
  name: string;
  angle?: Angle;
};

type BuildCreateStandardPayloadResult =
  | { ok: true; data: CreateStandardInput }
  | { ok: false; errors: Record<string, string> };

export const buildCreateStandardPayload = (
  values: CreateStandardFormValues
): BuildCreateStandardPayloadResult => {
  const parsed = createStandardSchema
    .superRefine((input, ctx) => {
      if (!input.groupId.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["groupId"],
          message: "Не удалось определить группу",
        });
      }

      if (!input.name.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["name"],
          message: "Название обязательно",
        });
      }

      if (!isAllowedAngle(input.angle)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["angle"],
          message: "Выберите корректный ракурс",
        });
      }
    })
    .transform((input) => ({
      groupId: input.groupId.trim(),
      name: input.name.trim(),
      angle: normalizeAngle(input.angle),
    }))
    .safeParse(values);

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

export const createStandard = ({
  groupId,
  name,
  angle,
}: CreateStandardInput): Promise<StandardMutationResponse> => {
  return client.post(`/standards`, { group_id: groupId, name, angle });
};

type Options = {
  mutationConfig?: MutationConfig<typeof createStandard>;
};

export const useCreateStandard = ({ mutationConfig }: Options = {}) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: createStandard,
    onSuccess: async (data, vars, ctx, mutation) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.groups.all() }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(vars.groupId) }),
      ]);
      notifySuccess("Эталон успешно создан");
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
