import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { notifySuccess, type MutationConfig } from "@/lib/react-query";
import type { Camera } from "@/types/contracts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { CameraFormValues, CameraPayloadBase } from "../lib/camera-form";
import {
  buildNormalizedCameraData,
  parseCameraFormValues,
  type BuildCameraPayloadResult,
} from "../lib/camera-form";

export type CameraCreatePayload = CameraPayloadBase & {
  password: string | null;
};

export const buildCreateCameraPayload = (
  values: CameraFormValues
): BuildCameraPayloadResult<CameraCreatePayload> => {
  const parsed = parseCameraFormValues(values);
  if (parsed.ok === false) {
    return parsed;
  }

  return {
    ok: true,
    data: {
      ...buildNormalizedCameraData(parsed.data),
      password: getCreatePasswordValue(parsed.data),
    },
  };
};

export const createCamera = (data: CameraCreatePayload): Promise<Camera> => {
  return client.post("/cameras", data);
};

type Options = {
  mutationConfig?: MutationConfig<typeof createCamera>;
};

export const useCreateCamera = ({ mutationConfig }: Options = {}) => {
  const qc = useQueryClient();
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: createCamera,
    onSuccess: async (data, vars, ctx, mutation) => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.cameras.all() }),
        qc.invalidateQueries({ queryKey: queryKeys.cameras.detail(data.id) }),
      ]);
      notifySuccess("Камера успешно создана");
      onSuccess?.(data, vars, ctx, mutation);
    },
    ...rest,
  });
};

const getCreatePasswordValue = (
  values: {
    has_auth: boolean;
    protocol: CameraPayloadBase["protocol"];
    password: string;
  }
) => {
  if (!values.has_auth || values.protocol === "usb") {
    return null;
  }

  const password = values.password.trim();
  return password === "" ? null : password;
};
