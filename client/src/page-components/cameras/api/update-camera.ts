import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { Camera } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { CameraFormValues } from "../lib/camera-form";
import {
  buildComparableCameraData,
  buildNormalizedCameraData,
  parseCameraFormValues,
  type BuildCameraPayloadResult,
} from "../lib/camera-form";
import type { CameraCreatePayload } from "./create-camera";

export type CameraUpdatePayload = Partial<CameraCreatePayload>;

export const buildUpdateCameraPayload = (
  current: CameraFormValues,
  initial: CameraFormValues
): BuildCameraPayloadResult<CameraUpdatePayload | null> => {
  const currentParsed = parseCameraFormValues(current);
  if (currentParsed.ok === false) {
    return currentParsed;
  }

  const currentData = buildNormalizedCameraData(currentParsed.data);
  const initialData = buildComparableCameraData(initial);

  const changedBase: Partial<typeof currentData> = {};

  for (const key of Object.keys(currentData) as (keyof typeof currentData)[]) {
    const nextValue = currentData[key];

    if (!Object.is(nextValue, initialData[key])) {
      Object.assign(changedBase, { [key]: nextValue });
    }
  }

  const changed: CameraUpdatePayload = { ...changedBase };

  if (!currentParsed.data.has_auth) {
    if (initial.has_auth) {
      changed.password = null;
    }
  } else {
    const password = currentParsed.data.password.trim();
    if (password) {
      changed.password = password;
    }
  }

  return {
    ok: true,
    data: Object.keys(changed).length > 0 ? changed : null,
  };
};

export type UpdateCameraInput = {
  id: string;
  data: CameraUpdatePayload;
};

export const updateCamera = ({ id, data }: UpdateCameraInput): Promise<Camera> => {
  return client.put(`/cameras/${id}`, data);
};

type Options = {
  mutationConfig?: MutationConfig<typeof updateCamera>;
};

export const useUpdateCamera = ({ mutationConfig }: Options = {}) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: updateCamera,
    onSuccess: async (data, vars, ctx, mutation) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.cameras.all() }),
        qc.invalidateQueries({ queryKey: queryKeys.cameras.detail(data.id) }),
      ]);
      notifySuccess("Камера успешно обновлена");
      onSuccess?.(data, vars, ctx, mutation);
    },
    ...rest,
  });
};
