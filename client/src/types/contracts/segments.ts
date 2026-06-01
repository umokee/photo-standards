export interface SegmentClass {
  id: string;
  group_id: string;
  class_group_id: string | null;
  name: string;
  hue: number;
}

export interface SegmentClassCategory {
  id: string;
  group_id: string;
  name: string;
  segment_classes: SegmentClass[];
}

export interface SegmentClassWithPoints extends SegmentClass {
  points: number[][][];
}

export interface SaveSegmentClassesResponse {
  group_id: string;
  categories: SegmentClassCategory[];
  ungrouped_classes: SegmentClass[];
}
