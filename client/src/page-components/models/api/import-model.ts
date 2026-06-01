import { getConstantsOrThrow } from "@/constants";
import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type {
  Architecture,
  ImportedClassMappingDraft,
  ModelImportStartResponse,
} from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";

const importModelMappingSchema = z
  .discriminatedUnion("mode", [
    z.object({
      mode: z.literal("existing"),
      native_key: z.string().trim().min(1, "Ключ класса обязателен"),
      segment_class_id: z.string().trim().min(1, "Выберите существующий класс"),
    }),
    z
      .object({
        mode: z.literal("new"),
        native_key: z.string().trim().min(1, "Ключ класса обязателен"),
        new_class_name: z.string().trim().min(1, "Укажите название нового класса"),
        new_class_hue: z.number().int("Hue должен быть целым числом"),
        new_class_group_id: z.string().trim().nullable().optional(),
      })
      .superRefine((value, ctx) => {
        const constants = getConstantsOrThrow();
        const hue = constants.segments.hue;

        if (value.new_class_hue < hue.min || value.new_class_hue > hue.max) {
          ctx.addIssue({
            code: "custom",
            path: ["new_class_hue"],
            message: `Hue должен быть от ${hue.min} до ${hue.max}`,
          });
        }
      }),
  ])
  .superRefine((value, ctx) => {
    if (value.mode === "existing" && !value.segment_class_id) {
      ctx.addIssue({
        code: "custom",
        path: ["segment_class_id"],
        message: "Выберите существующий класс",
      });
    }
  })
  .transform((value) => value as ImportedClassMappingDraft);

const importModelSchema = z
  .object({
    group_id: z.string().trim().min(1, "Группа обязательна"),
    architecture: z
      .string()
      .trim()
      .min(1, "Выберите архитектуру")
      .transform((value) => {
        return value as Architecture;
      }),
    imgsz: z.string().trim().min(1, "Укажите размер изображения").transform(Number),
    activate: z.boolean(),
    mappings: z
      .array(importModelMappingSchema)
      .min(1, "Выберите хотя бы один класс для импорта")
      .transform((value) => value as ImportedClassMappingDraft[]),
    weights: z.instanceof(File).nullable(),
  })
  .superRefine((values, ctx) => {
    const constants = getConstantsOrThrow();
    const training = constants.training;

    if (!training.architectures.values.includes(values.architecture)) {
      ctx.addIssue({
        code: "custom",
        path: ["architecture"],
        message: "Выберите корректную архитектуру",
      });
    }

    if (!Number.isInteger(values.imgsz)) {
      ctx.addIssue({
        code: "custom",
        path: ["imgsz"],
        message: "Размер изображения должен быть числом",
      });
    } else if (!training.image_size.values.includes(values.imgsz)) {
      ctx.addIssue({
        code: "custom",
        path: ["imgsz"],
        message: `Размер изображения должен быть одним из: ${training.image_size.values.join(", ")}`,
      });
    }

    if (!values.weights) {
      ctx.addIssue({
        code: "custom",
        path: ["weights"],
        message: "Выберите .pt файл",
      });
    } else if (!values.weights.name.toLowerCase().endsWith(".pt")) {
      ctx.addIssue({
        code: "custom",
        path: ["weights"],
        message: "Поддерживается только .pt файл",
      });
    }
  });

export type ImportModelFormValues = z.input<typeof importModelSchema>;
export type ImportModelPayload = Omit<z.output<typeof importModelSchema>, "weights"> & {
  weights: File;
};

type BuildImportModelPayloadResult =
  | { ok: true; data: ImportModelPayload }
  | { ok: false; errors: Record<string, string> };

export const buildImportModelPayload = (
  values: ImportModelFormValues
): BuildImportModelPayloadResult => {
  const parsed = importModelSchema.safeParse(values);

  if (!parsed.success) {
    const errors: Record<string, string> = {};

    for (const issue of parsed.error.issues) {
      const key = issue.path[0] === "mappings" ? "mappings" : String(issue.path[0] ?? "form");
      if (!errors[key]) {
        errors[key] = issue.message;
      }
    }

    return {
      ok: false,
      errors,
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

export const importModel = async (input: ImportModelPayload): Promise<ModelImportStartResponse> => {
  const formData = new FormData();

  formData.append("group_id", input.group_id);
  formData.append("architecture", input.architecture);
  formData.append("imgsz", String(input.imgsz));
  formData.append("activate", String(input.activate));
  formData.append("mappings_json", JSON.stringify(input.mappings));
  formData.append("weights", input.weights);

  return client.post("/yolo/import", formData);
};

type Options = {
  groupId: string;
  mutationConfig?: MutationConfig<typeof importModel>;
};

export const useImportModel = ({ groupId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: importModel,
    onSuccess: async (data, vars, ctx, mutationCtx) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.training.models(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.model(data.model_id) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.tasks(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.task(data.task_id) }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
      ]);

      notifySuccess("Импорт модели поставлен в очередь");
      onSuccess?.(data, vars, ctx, mutationCtx);
    },
    ...rest,
  });
};
