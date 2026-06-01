import { useEffect, useReducer, useState } from "react";

import { GroupDetail, SegmentClass, SegmentClassCategory } from "@/types/contracts";
import { useSaveSegmentClasses } from "../api/save-segment-classes";

interface ClassState {
  key: string;
  id: string | null;
  name: string;
  hue: number;
}

interface CategoryState {
  key: string;
  id: string | null;
  name: string;
  collapsed: boolean;
  segmentClasses: ClassState[];
}

interface EditorState {
  categories: CategoryState[];
  ungroupedClasses: ClassState[];
  deletedCategoryIds: Set<string>;
  deletedClassIds: Set<string>;
}

type Action =
  | { type: "reset"; payload: EditorState }
  | { type: "category/add" }
  | { type: "category/remove"; key: string }
  | { type: "category/toggle"; key: string }
  | { type: "category/update-name"; key: string; value: string }
  | { type: "class/add-to-category"; categoryKey: string }
  | { type: "class/add-ungrouped" }
  | { type: "class/remove-from-category"; categoryKey: string; classKey: string }
  | { type: "class/remove-ungrouped"; classKey: string }
  | {
      type: "class/update-name";
      categoryKey: string | null;
      classKey: string;
      value: string;
    }
  | {
      type: "class/update-hue";
      categoryKey: string | null;
      classKey: string;
      value: number;
    };

const uid = () => crypto.randomUUID();

const createEmptyClass = (): ClassState => ({
  key: uid(),
  id: null,
  name: "",
  hue: 210,
});

const createEmptyCategory = (): CategoryState => ({
  key: uid(),
  id: null,
  name: "",
  collapsed: false,
  segmentClasses: [],
});

const mapClass = (item: SegmentClass): ClassState => ({
  key: item.id,
  id: item.id,
  name: item.name,
  hue: item.hue,
});

const mapCategory = (item: SegmentClassCategory): CategoryState => ({
  key: item.id,
  id: item.id,
  name: item.name,
  collapsed: true,
  segmentClasses: item.segment_classes.map(mapClass),
});

const createInitialState = (group: GroupDetail): EditorState => ({
  categories: group.segment_class_categories.map(mapCategory),
  ungroupedClasses: group.ungrouped_segment_classes.map(mapClass),
  deletedCategoryIds: new Set<string>(),
  deletedClassIds: new Set<string>(),
});

const serializeClasses = (items: ClassState[]) =>
  items
    .filter((item) => item.name.trim())
    .map((item) => ({
      id: item.id ?? undefined,
      name: item.name.trim(),
      hue: item.hue,
    }));

const serializeCategories = (items: CategoryState[]) =>
  items
    .filter((item) => item.name.trim())
    .map((item) => ({
      id: item.id ?? undefined,
      name: item.name.trim(),
      segment_classes: serializeClasses(item.segmentClasses),
    }));

const reducer = (state: EditorState, action: Action): EditorState => {
  switch (action.type) {
    case "reset":
      return action.payload;

    case "category/add":
      return {
        ...state,
        categories: [...state.categories, createEmptyCategory()],
      };

    case "category/remove": {
      const category = state.categories.find((item) => item.key === action.key);
      if (!category) return state;

      const nextDeletedCategoryIds = new Set(state.deletedCategoryIds);
      if (category.id) {
        nextDeletedCategoryIds.add(category.id);
      }

      return {
        ...state,
        categories: state.categories.filter((item) => item.key !== action.key),
        ungroupedClasses: [...state.ungroupedClasses, ...category.segmentClasses],
        deletedCategoryIds: nextDeletedCategoryIds,
      };
    }

    case "category/toggle":
      return {
        ...state,
        categories: state.categories.map((item) =>
          item.key === action.key ? { ...item, collapsed: !item.collapsed } : item
        ),
      };

    case "category/update-name":
      return {
        ...state,
        categories: state.categories.map((item) =>
          item.key === action.key ? { ...item, name: action.value } : item
        ),
      };

    case "class/add-to-category":
      return {
        ...state,
        categories: state.categories.map((item) =>
          item.key === action.categoryKey
            ? {
                ...item,
                collapsed: false,
                segmentClasses: [...item.segmentClasses, createEmptyClass()],
              }
            : item
        ),
      };

    case "class/add-ungrouped":
      return {
        ...state,
        ungroupedClasses: [...state.ungroupedClasses, createEmptyClass()],
      };

    case "class/remove-from-category": {
      const category = state.categories.find((item) => item.key === action.categoryKey);
      const target = category?.segmentClasses.find((item) => item.key === action.classKey);
      if (!target) return state;

      const nextDeletedClassIds = new Set(state.deletedClassIds);
      if (target.id) {
        nextDeletedClassIds.add(target.id);
      }

      return {
        ...state,
        categories: state.categories.map((item) =>
          item.key === action.categoryKey
            ? {
                ...item,
                segmentClasses: item.segmentClasses.filter(
                  (segmentClass) => segmentClass.key !== action.classKey
                ),
              }
            : item
        ),
        deletedClassIds: nextDeletedClassIds,
      };
    }

    case "class/remove-ungrouped": {
      const target = state.ungroupedClasses.find((item) => item.key === action.classKey);
      if (!target) return state;

      const nextDeletedClassIds = new Set(state.deletedClassIds);
      if (target.id) {
        nextDeletedClassIds.add(target.id);
      }

      return {
        ...state,
        ungroupedClasses: state.ungroupedClasses.filter((item) => item.key !== action.classKey),
        deletedClassIds: nextDeletedClassIds,
      };
    }

    case "class/update-name":
      if (action.categoryKey === null) {
        return {
          ...state,
          ungroupedClasses: state.ungroupedClasses.map((item) =>
            item.key === action.classKey ? { ...item, name: action.value } : item
          ),
        };
      }

      return {
        ...state,
        categories: state.categories.map((item) =>
          item.key === action.categoryKey
            ? {
                ...item,
                segmentClasses: item.segmentClasses.map((segmentClass) =>
                  segmentClass.key === action.classKey
                    ? { ...segmentClass, name: action.value }
                    : segmentClass
                ),
              }
            : item
        ),
      };

    case "class/update-hue":
      if (action.categoryKey === null) {
        return {
          ...state,
          ungroupedClasses: state.ungroupedClasses.map((item) =>
            item.key === action.classKey ? { ...item, hue: action.value } : item
          ),
        };
      }

      return {
        ...state,
        categories: state.categories.map((item) =>
          item.key === action.categoryKey
            ? {
                ...item,
                segmentClasses: item.segmentClasses.map((segmentClass) =>
                  segmentClass.key === action.classKey
                    ? { ...segmentClass, hue: action.value }
                    : segmentClass
                ),
              }
            : item
        ),
      };

    default:
      return state;
  }
};

export const useManageSegmentGroups = (group: GroupDetail) => {
  const [state, dispatch] = useReducer(reducer, group, createInitialState);
  const [activeColorKey, setActiveColorKey] = useState<string | null>(null);

  useEffect(() => {
    dispatch({ type: "reset", payload: createInitialState(group) });
    setActiveColorKey(null);
  }, [group]);

  const mutation = useSaveSegmentClasses({
    groupId: group.id,
  });

  const save = async () => {
    try {
      await mutation.mutateAsync({
        groupId: group.id,
        categories: serializeCategories(state.categories),
        ungroupedClasses: serializeClasses(state.ungroupedClasses),
        deletedCategoryIds: [...state.deletedCategoryIds],
        deletedClassIds: [...state.deletedClassIds],
      });

      return true;
    } catch {
      return false;
    }
  };

  return {
    categories: state.categories,
    ungroupedClasses: state.ungroupedClasses,
    saving: mutation.isPending,
    activeColorKey,
    toggleColorPicker: (key: string) => setActiveColorKey((prev) => (prev === key ? null : key)),
    closeColorPicker: () => setActiveColorKey(null),

    categoryActions: {
      add: () => dispatch({ type: "category/add" }),
      remove: (key: string) => dispatch({ type: "category/remove", key }),
      toggle: (key: string) => dispatch({ type: "category/toggle", key }),
      updateName: (key: string, value: string) =>
        dispatch({ type: "category/update-name", key, value }),
    },

    classActions: {
      addToCategory: (categoryKey: string) =>
        dispatch({ type: "class/add-to-category", categoryKey }),
      addUngrouped: () => dispatch({ type: "class/add-ungrouped" }),

      removeFromCategory: (categoryKey: string, classKey: string) =>
        dispatch({ type: "class/remove-from-category", categoryKey, classKey }),

      removeUngrouped: (classKey: string) => dispatch({ type: "class/remove-ungrouped", classKey }),

      updateName: (categoryKey: string | null, classKey: string, value: string) =>
        dispatch({
          type: "class/update-name",
          categoryKey,
          classKey,
          value,
        }),

      updateHue: (categoryKey: string | null, classKey: string, value: number) =>
        dispatch({
          type: "class/update-hue",
          categoryKey,
          classKey,
          value,
        }),
    },

    save,
  };
};
