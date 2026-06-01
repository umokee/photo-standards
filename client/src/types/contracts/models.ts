import { Architecture } from "./shared";

export type ImportedClassMatchReason = "uuid" | "name";

export interface MlModelClassMeta {
  id: string;
  key: string;
  name: string;
  index: number;
  class_group_id: string | null;
  hue?: number | null;
  native_key?: string | null;
  native_index?: number | null;
  source?: string | null;
}

export interface MlModel {
  id: string;
  group_id: string;

  architecture: Architecture;
  weights_path: string | null;
  version: number | null;

  epochs: number | null;
  imgsz: number;
  batch_size: number | null;

  num_classes: number | null;
  class_keys: string[] | null;
  class_meta: MlModelClassMeta[] | null;
  metrics: Record<string, number | null> | null;

  train_ratio: number | null;
  val_ratio: number | null;
  test_ratio: number | null;

  total_images: number | null;
  train_count: number | null;
  val_count: number | null;
  test_count: number | null;

  is_active: boolean;
  trained_at: string | null;
  created_at: string;
}

export interface ImportedClassSuggestion {
  segment_class_id: string;
  segment_class_name: string;
  match_reason: ImportedClassMatchReason;
}

export interface ImportedNativeClass {
  index: number;
  key: string;
  suggested: ImportedClassSuggestion | null;
}

export interface ModelImportPreviewResponse {
  native_classes: ImportedNativeClass[];
}

export interface ExistingImportedClassMappingDraft {
  mode: "existing";
  native_key: string;
  segment_class_id: string;
}

export interface NewImportedClassMappingDraft {
  mode: "new";
  native_key: string;
  new_class_name: string;
  new_class_hue: number;
  new_class_group_id?: string | null;
}

export type ImportedClassMappingDraft =
  | ExistingImportedClassMappingDraft
  | NewImportedClassMappingDraft;

export interface ModelImportStartResponse {
  task_id: string;
  model_id: string;
}
