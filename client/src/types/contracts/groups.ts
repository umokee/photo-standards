import { MlModelClassMeta } from "./models";
import { SegmentClass, SegmentClassCategory } from "./segments";
import { Architecture } from "./shared";

export interface GroupStats {
  standards_count: number;
  images_count: number;
  annotated_images_count: number;
  polygons_count: number;
  segment_class_groups_count: number;
  segment_classes_count: number;
  models_count: number;
  inspections_count: number;
}

export interface GroupStandard {
  id: string;
  group_id: string;
  name: string;
  angle: string | null;
  is_active: boolean;
  created_at: string;
  reference_path: string | null;
  images_count: number;
  annotated_images_count: number;
}

export interface GroupModel {
  id: string;
  group_id: string;
  architecture: Architecture;
  version: number | null;
  epochs: number | null;
  imgsz: number;
  batch_size: number | null;
  num_classes: number | null;
  class_keys: string[] | null;
  class_meta: MlModelClassMeta[] | null;
  metrics: Record<string, number | null> | null;
  is_active: boolean;
  trained_at: string | null;
  created_at: string;
}

export interface GroupListItem {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  stats: GroupStats;
}

export interface GroupDetail {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  stats: GroupStats;
  standards: GroupStandard[];
  active_model: GroupModel | null;
  segment_class_categories: SegmentClassCategory[];
  ungrouped_segment_classes: SegmentClass[];
}

export interface GroupMutationResponse {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
}
