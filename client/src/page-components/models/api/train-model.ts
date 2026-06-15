import { getConstantsOrThrow } from "@/constants";
import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import { Architecture, TrainingStartResponse } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";

const trainModelSchema = z
  .object({
    group_id: z.string().trim().min(1, "Выберите группу"),
    architecture: z
      .string()
      .trim()
      .min(1, "Выберите архитектуру")
      .transform((val) => {
        return val as Architecture;
      }),
    train_ratio: z.string().trim().min(1, "Укажите долю train, %").transform(Number),
    val_ratio: z.string().trim().min(1, "Укажите долю val, %").transform(Number),
    epochs: z.string().trim().min(1, "Укажите количество эпох").transform(Number),
    imgsz: z.string().trim().min(1, "Укажите размер изображения").transform(Number),
    batch_size: z.string().trim().min(1, "Укажите размер батча").transform(Number),
  })
  .superRefine((values, ctx) => {
    const constants = getConstantsOrThrow();
    const training = constants.training;
    const safeRatioSumMax = Math.min(training.ratio_sum_max, 100);

    if (!training.architectures.values.includes(values.architecture)) {
      ctx.addIssue({
        code: "custom",
        path: ["architecture"],
        message: "Выберите архитектуру из списка",
      });
    }

    if (!Number.isInteger(values.epochs)) {
      ctx.addIssue({
        code: "custom",
        path: ["epochs"],
        message: "Эпохи должны быть целым числом",
      });
    } else if (values.epochs < training.epochs.min || values.epochs > training.epochs.max) {
      ctx.addIssue({
        code: "custom",
        path: ["epochs"],
        message: `Эпохи должны быть от ${training.epochs.min} до ${training.epochs.max}`,
      });
    }

    if (!Number.isInteger(values.batch_size)) {
      ctx.addIssue({
        code: "custom",
        path: ["batch_size"],
        message: "Размер батча должен быть целым числом",
      });
    } else if (
      values.batch_size < training.batch_size.min ||
      values.batch_size > training.batch_size.max
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["batch_size"],
        message: `Размер батча должен быть от ${training.batch_size.min} до ${training.batch_size.max}`,
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

    if (!Number.isInteger(values.train_ratio)) {
      ctx.addIssue({
        code: "custom",
        path: ["train_ratio"],
        message: "Доля train должна быть целым числом",
      });
    } else if (
      values.train_ratio < training.train_ratio.min ||
      values.train_ratio > training.train_ratio.max
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["train_ratio"],
        message: `Доля train должна быть от ${training.train_ratio.min} до ${training.train_ratio.max}`,
      });
    }

    if (!Number.isInteger(values.val_ratio)) {
      ctx.addIssue({
        code: "custom",
        path: ["val_ratio"],
        message: "Доля val должна быть целым числом",
      });
    } else if (
      values.val_ratio < training.val_ratio.min ||
      values.val_ratio > training.val_ratio.max
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["val_ratio"],
        message: `Доля val должна быть от ${training.val_ratio.min} до ${training.val_ratio.max}`,
      });
    }

    if (
      Number.isInteger(values.train_ratio) &&
      Number.isInteger(values.val_ratio) &&
      values.train_ratio + values.val_ratio > safeRatioSumMax
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["val_ratio"],
        message: `Сумма долей train и val должна быть не больше ${safeRatioSumMax}%`,
      });
    }
  });

export type TrainModelFormValues = z.input<typeof trainModelSchema>;
export type TrainModelPayload = z.output<typeof trainModelSchema>;

type BuildTrainModelPayloadResult =
  | { ok: true; data: TrainModelPayload }
  | { ok: false; errors: Record<string, string> };

export const buildTrainModelPayload = (
  values: TrainModelFormValues
): BuildTrainModelPayloadResult => {
  const parsed = trainModelSchema.safeParse(values);

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

export const trainModel = (data: TrainModelPayload): Promise<TrainingStartResponse> => {
  return client.post("/yolo/training/run", data);
};

type Options = {
  groupId: string;
  mutationConfig?: MutationConfig<typeof trainModel>;
};

export const useTrainModel = ({ groupId, mutationConfig }: Options) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: trainModel,
    onSuccess: async (data, vars, ctx, mutationCtx) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.training.models(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.model(data.model_id) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.tasks(groupId) }),
        qc.invalidateQueries({ queryKey: queryKeys.training.task(data.task_id) }),
        qc.invalidateQueries({ queryKey: queryKeys.groups.detail(groupId) }),
      ]);

      notifySuccess("Обучение модели запущено");
      onSuccess?.(data, vars, ctx, mutationCtx);
    },
    ...rest,
  });
};
