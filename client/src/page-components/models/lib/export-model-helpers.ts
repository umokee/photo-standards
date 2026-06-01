import type { MlModel } from "@/types/contracts";
import { architectureLabel } from "@/utils/labels";
import { getModelVersionLabel } from "./model-helpers";

export const isExportableModel = (model: MlModel): boolean => {
  return model.version !== null && !!model.weights_path;
};

export const sortModelsForExport = (a: MlModel, b: MlModel): number => {
  if (a.is_active !== b.is_active) {
    return a.is_active ? -1 : 1;
  }

  if ((a.version ?? 0) !== (b.version ?? 0)) {
    return (b.version ?? 0) - (a.version ?? 0);
  }

  return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
};

export const formatExportModelOptionLabel = (model: MlModel): string => {
  const versionLabel = getModelVersionLabel(model);
  return `${architectureLabel(model.architecture)} · ${versionLabel}${
    model.is_active ? " · активная" : ""
  }`;
};

export const buildExportModelFileName = (model: MlModel): string => {
  const version = model.version ?? "draft";
  return `yolo-${model.architecture}-v${version}.zip`;
};
