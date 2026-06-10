import { inspectionModePaths } from "@/app/paths";
import { client } from "@/lib/api-client";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { InspectionMode, InspectionStartResponse } from "@/types/contracts";
import { useMutation } from "@tanstack/react-query";
import { z } from "zod";
import { isDeviceCameraId } from "../lib/device-camera";

const runInspectionSchema = z
  .object({
    standard_id: z.string().trim().min(1, "Выберите эталон"),
    selected_segment_class_ids: z
      .array(z.string().trim().min(1))
      .transform((values) => values.filter(Boolean)),
    mode: z
      .string()
      .trim()
      .refine(
        (value): value is InspectionMode =>
          inspectionModePaths.includes(value as (typeof inspectionModePaths)[number]),
        "Выберите корректный режим проверки"
      ),
    image: z.instanceof(File).nullable().optional(),
    camera_id: z.string().trim().nullable().optional(),
    notes: z.string().nullable().optional(),
  })
  .superRefine((values, ctx) => {
    if (values.selected_segment_class_ids.length === 0) {
      ctx.addIssue({
        code: "custom",
        path: ["selected_segment_class_ids"],
        message: "Выберите хотя бы один элемент контроля",
      });
    }

    if (values.mode === "photo" && !values.image) {
      ctx.addIssue({
        code: "custom",
        path: ["image"],
        message: "Добавьте изображение",
      });
    }

    const cameraId = values.camera_id?.trim() ?? "";
    const hasDeviceCamera = isDeviceCameraId(cameraId);

    if (values.mode === "snapshot" && !values.image) {
      if (!cameraId) {
        ctx.addIssue({
          code: "custom",
          path: ["camera_id"],
          message: "Выберите камеру",
        });
      }

      if (hasDeviceCamera) {
        ctx.addIssue({
          code: "custom",
          path: ["image"],
          message: "Не удалось получить кадр камеры устройства",
        });
      }
    }

    if (values.mode === "realtime" && !cameraId && !hasDeviceCamera) {
      ctx.addIssue({
        code: "custom",
        path: ["camera_id"],
        message: "Выберите камеру",
      });
    }
  })
  .transform((values) => ({
    standard_id: values.standard_id,
    selected_segment_class_ids: values.selected_segment_class_ids,
    mode: values.mode,
    image: values.image ?? undefined,
    camera_id:
      values.camera_id && !isDeviceCameraId(values.camera_id) ? values.camera_id.trim() : undefined,
    notes: values.notes?.trim() || null,
  }));

export type RunInspectionFormValues = z.input<typeof runInspectionSchema>;
export type RunInspectionInput = z.output<typeof runInspectionSchema>;

type BuildRunInspectionPayloadResult =
  | { ok: true; data: RunInspectionInput }
  | { ok: false; errors: Record<string, string> };

export const buildRunInspectionPayload = (
  values: RunInspectionFormValues
): BuildRunInspectionPayloadResult => {
  const parsed = runInspectionSchema.safeParse(values);

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

export const runInspection = async (
  input: RunInspectionInput
): Promise<InspectionStartResponse> => {
  const formData = new FormData();

  formData.append("standard_id", input.standard_id);
  formData.append("mode", input.mode);

  input.selected_segment_class_ids.forEach((id) => {
    formData.append("selected_segment_class_ids", id);
  });

  if (input.image) {
    formData.append("image", input.image);
  }

  if (input.camera_id) {
    formData.append("camera_id", input.camera_id);
  }

  if (input.notes) {
    formData.append("notes", input.notes);
  }

  return client.post("/yolo/inspection/run", formData);
};

type Options = {
  mutationConfig?: MutationConfig<typeof runInspection>;
};

export const useRunInspection = ({ mutationConfig }: Options = {}) => {
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: runInspection,
    onSuccess: (...args) => {
      notifySuccess("Проверка запущена");
      onSuccess?.(...args);
    },
    ...rest,
  });
};
