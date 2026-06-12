import type { GroupDetail, SegmentClass, SegmentClassCategory, StandardDetail } from "@/types/contracts";

type ClassSelectorSource = GroupDetail | StandardDetail;

function getClassCollections(source: ClassSelectorSource): {
  categories: SegmentClassCategory[];
  ungrouped: SegmentClass[];
} {
  if ("used_segment_class_categories" in source || "used_ungrouped_segment_classes" in source) {
    return {
      categories: source.used_segment_class_categories ?? source.segment_class_categories,
      ungrouped: source.used_ungrouped_segment_classes ?? source.ungrouped_segment_classes,
    };
  }

  return {
    categories: source.segment_class_categories,
    ungrouped: source.ungrouped_segment_classes,
  };
}

export type ClassSelectorGroup = {
  id: string;
  name: string;
  items: {
    id: string;
    name: string;
    hue: number | null;
  }[];
};

export const buildClassSelectorGroups = (source: ClassSelectorSource): ClassSelectorGroup[] => {
  const { categories, ungrouped } = getClassCollections(source);

  const categoryGroups = categories.map((category) => ({
    id: category.id,
    name: category.name,
    items: category.segment_classes.map((item) => ({
      id: item.id,
      name: item.name,
      hue: item.hue,
    })),
  }));

  if (ungrouped.length > 0) {
    categoryGroups.push({
      id: "ungrouped",
      name: "Без категории",
      items: ungrouped.map((item) => ({
        id: item.id,
        name: item.name,
        hue: item.hue,
      })),
    });
  }

  return categoryGroups;
};

export const getClassSelectorAllIds = (source: ClassSelectorSource): string[] => {
  return buildClassSelectorGroups(source).flatMap((groupItem) =>
    groupItem.items.map((item) => item.id)
  );
};
