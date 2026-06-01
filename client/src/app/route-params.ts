import { redirect } from "react-router-dom";
import { InspectionModePath, inspectionModePaths, paths } from "./paths";

export function requireParam(value: string | undefined, fallbackPath: string): string {
  const normalized = value?.trim();

  if (!normalized) {
    throw redirect(fallbackPath);
  }

  return normalized;
}

export function requireInspectionMode(value: string | undefined): InspectionModePath {
  const normalized = value?.trim();

  if (!normalized) {
    throw redirect(paths.inspectionMode("photo"));
  }

  const mode = inspectionModePaths.find((item) => item === normalized);

  if (!mode) {
    throw redirect(paths.inspectionMode("photo"));
  }

  return mode;
}
