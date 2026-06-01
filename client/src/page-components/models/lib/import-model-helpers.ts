import type {
  GroupDetail,
  ImportedClassMatchReason,
  ImportedNativeClass,
} from "@/types/contracts";

export type ImportModelMappingMode = "ignore" | "existing" | "new";

export type ImportModelMappingRow = {
  nativeKey: string;
  nativeIndex: number;
  mode: ImportModelMappingMode;
  segmentClassId: string;
  newClassName: string;
  newClassHue: string;
  newClassGroupId: string;
  suggestedSegmentClassName: string | null;
  suggestedMatchReason: ImportedClassMatchReason | null;
};

export type ImportModelSelectOption = {
  value: string;
  label: string;
};

export const buildImportModelExistingClassOptions = (
  group: GroupDetail | undefined
): ImportModelSelectOption[] => {
  if (!group) return [];

  const categoryItems = group.segment_class_categories.flatMap((category) =>
    category.segment_classes.map((item) => ({
      value: item.id,
      label: `${category.name} / ${item.name}`,
    }))
  );

  const ungroupedItems = group.ungrouped_segment_classes.map((item) => ({
    value: item.id,
    label: `Без категории / ${item.name}`,
  }));

  return [...categoryItems, ...ungroupedItems];
};

export const buildImportModelCategoryOptions = (
  group: GroupDetail | undefined
): ImportModelSelectOption[] => {
  if (!group) {
    return [{ value: "", label: "Без категории" }];
  }

  return [
    { value: "", label: "Без категории" },
    ...group.segment_class_categories.map((item) => ({
      value: item.id,
      label: item.name,
    })),
  ];
};

export const buildImportModelPreviewRows = (
  nativeClasses: ImportedNativeClass[],
  group: GroupDetail | undefined,
  defaultHue: number
): ImportModelMappingRow[] => {
  const classById = buildGroupClassesById(group);

  return nativeClasses.map((nativeItem) => {
    const suggestedId = nativeItem.suggested?.segment_class_id;
    const suggestedClass = suggestedId ? classById[suggestedId] : null;

    if (suggestedId && suggestedClass) {
      return {
        nativeKey: nativeItem.key,
        nativeIndex: nativeItem.index,
        mode: "existing",
        segmentClassId: suggestedClass.id,
        newClassName: "",
        newClassHue: String(suggestedClass.hue),
        newClassGroupId: suggestedClass.class_group_id ?? "",
        suggestedSegmentClassName: nativeItem.suggested?.segment_class_name ?? suggestedClass.name,
        suggestedMatchReason: nativeItem.suggested?.match_reason ?? null,
      };
    }

    return {
      nativeKey: nativeItem.key,
      nativeIndex: nativeItem.index,
      mode: "ignore",
      segmentClassId: "",
      newClassName: isUuidLike(nativeItem.key) ? "" : nativeItem.key,
      newClassHue: String(defaultHue),
      newClassGroupId: "",
      suggestedSegmentClassName: nativeItem.suggested?.segment_class_name ?? null,
      suggestedMatchReason: nativeItem.suggested?.match_reason ?? null,
    };
  });
};

export const isImportModelMappingRowValid = (row: ImportModelMappingRow): boolean => {
  if (row.mode === "ignore") return true;
  if (row.mode === "existing") return !!row.segmentClassId;
  return !!row.newClassName.trim() && row.newClassHue !== "";
};

type GroupClassInfo = {
  id: string;
  name: string;
  hue: number;
  class_group_id: string | null;
};

function buildGroupClassesById(group: GroupDetail | undefined): Record<string, GroupClassInfo> {
  if (!group) return {};

  const allClasses = [
    ...group.segment_class_categories.flatMap((category) => category.segment_classes),
    ...group.ungrouped_segment_classes,
  ];

  return Object.fromEntries(
    allClasses.map((item) => [
      item.id,
      {
        id: item.id,
        name: item.name,
        hue: item.hue,
        class_group_id: item.class_group_id,
      },
    ])
  );
}

function isUuidLike(value: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
    value.trim()
  );
}
