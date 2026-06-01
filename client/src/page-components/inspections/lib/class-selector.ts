import type { GroupDetail } from "@/types/contracts";

export type ClassSelectorGroup = {
  id: string;
  name: string;
  items: {
    id: string;
    name: string;
    hue: number | null;
  }[];
};

export const buildClassSelectorGroups = (group: GroupDetail): ClassSelectorGroup[] => {
  const categoryGroups = group.segment_class_categories.map((category) => ({
    id: category.id,
    name: category.name,
    items: category.segment_classes.map((item) => ({
      id: item.id,
      name: item.name,
      hue: item.hue,
    })),
  }));

  if (group.ungrouped_segment_classes.length > 0) {
    categoryGroups.push({
      id: "ungrouped",
      name: "Без категории",
      items: group.ungrouped_segment_classes.map((item) => ({
        id: item.id,
        name: item.name,
        hue: item.hue,
      })),
    });
  }

  return categoryGroups;
};

export const getClassSelectorAllIds = (group: GroupDetail): string[] => {
  return buildClassSelectorGroups(group).flatMap((groupItem) =>
    groupItem.items.map((item) => item.id)
  );
};
