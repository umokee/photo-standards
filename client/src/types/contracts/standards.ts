import type { SegmentClass, SegmentClassCategory, SegmentClassWithPoints } from "./segments";
import type { Angle } from "./shared";

export interface StandardStats {
  images_count: number;
  annotated_images_count: number;
  unannotated_images_count: number;
  segment_classes_count: number;
  segment_class_categories_count: number;
  reference_image_id: string | null;
  reference_path: string | null;
}

export interface StandardImage {
  id: string;
  standard_id: string;
  image_path: string;
  is_reference: boolean;
  annotation_count: number;
  created_at: string;
}

export interface StandardImageDetail extends StandardImage {
  segment_classes: SegmentClassWithPoints[];
}

export interface StandardMutationResponse {
  id: string;
  group_id: string;
  name: string;
  angle: Angle | null;
  is_active: boolean;
  created_at: string;
}

export interface StandardDetail {
  id: string;
  group_id: string;
  name: string;
  angle: Angle | null;
  is_active: boolean;
  created_at: string;
  stats: StandardStats;
  images: StandardImage[];
  segment_class_categories: SegmentClassCategory[];
  ungrouped_segment_classes: SegmentClass[];
}
