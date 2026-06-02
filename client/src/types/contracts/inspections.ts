import { InspectionMode, InspectionStatus } from "./shared";

export interface InspectionStartResponse {
  kind: "task" | "session";
  task_id: string | null;
  session_id: string | null;
  status: string;
  message: string;
}

export interface InspectionDetectionBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type InspectionSegmentStatus = "ok" | "missing" | "extra" | "unmatched";

export type InspectionAlignmentStatus =
  | "success"
  | "insufficient_matches"
  | "insufficient_inliers"
  | "homography_failed";

export interface InspectionTaskSegmentDetail {
  annotation_id: string | null;
  segment_class_id: string | null;
  class_key: string;
  name: string;
  hue: number | null;

  status: InspectionSegmentStatus;
  iou: number | null;
  confidence: number | null;

  expected_polygon: number[][] | null;
  detected_polygon: number[][] | null;
  detected_bbox: InspectionDetectionBBox | null;
  debug?: Record<string, unknown> | null;
}

export interface InspectionTaskResult {
  task_id: string;
  inspection_id: string | null;
  status: InspectionStatus;
  passed: boolean;
  matched: number;
  total: number;
  missing: string[];

  alignment_status: InspectionAlignmentStatus | null;
  alignment_inlier_count: number | null;
  alignment_raw_match_count: number | null;
  homography: number[][] | null;

  details: InspectionTaskSegmentDetail[];
  mode: InspectionMode;
  model_name: string | null;
  image_path: string;
  result_image_path: string | null;
  debug?: {
    model_class_keys?: string[];
    raw_counts?: Record<string, number>;
    imgsz?: number;
  } | null;
}

export interface InspectionRealtimeStatus {
  state: "warming_up" | "online" | "failed";
  matched: number;
  total: number;
  missing: string[];
  status: InspectionStatus;
  passed: boolean;
  alignment_status: string;
  alignment_inlier_count: number | null;
  alignment_raw_match_count: number | null;
  captured_at: string | null;
  details: InspectionTaskSegmentDetail[];
}

export interface InspectionSaveResponse {
  inspection_id: string;
  status: InspectionStatus;
  message: string;
}

export interface InspectionHistoryItem {
  id: string;
  group_id: string | null;
  standard_id: string | null;
  standard_name: string | null;
  standard_reference_path: string | null;
  model_id: string | null;
  model_name: string | null;
  camera_id: string | null;
  camera_name: string | null;
  mode: InspectionMode;
  status: InspectionStatus;
  image_path: string;
  result_image_path: string | null;
  total_segments: number;
  matched_segments: number;
  notes: string | null;
  inspected_at: string;
}

export interface InspectionSegmentResultItem {
  id: string;
  segment_annotation_id: string | null;
  segment_class_id: string | null;
  class_key: string;
  name: string;
  hue: number | null;

  status: InspectionSegmentStatus;
  iou: number | null;
  confidence: number | null;

  expected_polygon: number[][] | null;
  detected_polygon: number[][] | null;
  detected_bbox: InspectionDetectionBBox | null;
  debug?: Record<string, unknown> | null;
}

export interface InspectionResult {
  id: string;
  group_id: string | null;
  standard_id: string | null;
  standard_name: string | null;
  standard_reference_path: string | null;
  model_id: string | null;
  model_name: string | null;
  camera_id: string | null;
  camera_name: string | null;
  user_id: string | null;
  user_name: string | null;
  image_path: string;
  result_image_path: string | null;
  status: InspectionStatus;
  mode: InspectionMode;
  total_segments: number;
  matched_segments: number;

  alignment_status: InspectionAlignmentStatus | null;
  alignment_inlier_count: number | null;
  alignment_raw_match_count: number | null;
  homography: number[][] | null;

  notes: string | null;
  debug_payload: Record<string, unknown> | null;
  inspected_at: string;

  segment_results: InspectionSegmentResultItem[];
}
