import { angleLabel, architectureLabel, inspectionModeLabel, roleLabel } from "@/utils/labels";
import { useMemo } from "react";
import { queryClient } from "./lib/query-client";
import { queryKeys } from "./lib/query-keys";
import { AppConstantsResponse, useGetConstants } from "./page-components/meta/get-constants";

const getCachedConstants = (): AppConstantsResponse | null => {
  return queryClient.getQueryData<AppConstantsResponse>(queryKeys.meta.constants()) ?? null;
};

export const getConstantsOrThrow = (): AppConstantsResponse => {
  const data = getCachedConstants();
  if (!data) {
    throw new Error("Константы не загружены");
  }
  return data;
};

export const useAppConstants = () => {
  const query = useGetConstants();

  if (!query.data) {
    throw new Error("Константы не загружены");
  }

  return query.data;
};

export const useRoleOptions = () => {
  const data = useAppConstants();

  return useMemo(
    () =>
      data.users.roles.values.map((value) => ({
        value,
        label: roleLabel(value),
      })),
    [data]
  );
};

export const useAngleOptions = () => {
  const data = useAppConstants();

  return useMemo(
    () =>
      data.standards.angles.values.map((value) => ({
        value,
        label: angleLabel(value),
      })),
    [data]
  );
};

export const useInspectionModeOptions = () => {
  const data = useAppConstants();

  return useMemo(
    () =>
      data.inspections.modes.values.map((value) => ({
        value,
        label: inspectionModeLabel(value),
      })),
    [data]
  );
};

export const useArchitectureOptions = () => {
  const data = useAppConstants();

  return useMemo(
    () =>
      data.training.architectures.values.map((value) => ({
        value,
        label: architectureLabel(value),
      })),
    [data]
  );
};

export const useImageSizeOptions = () => {
  const data = useAppConstants();

  return useMemo(
    () =>
      data.training.image_size.values.map((value) => ({
        value: String(value),
        label: String(value),
      })),
    [data]
  );
};

export const useTrainingLimits = () => {
  const data = useAppConstants();

  return useMemo(
    () => ({
      epochs: data.training.epochs,
      batch_size: data.training.batch_size,
      train_ratio: data.training.train_ratio,
      val_ratio: data.training.val_ratio,
      ratio_sum_max: data.training.ratio_sum_max,
      min_images_to_train: data.training.min_images_to_train,
    }),
    [data]
  );
};
