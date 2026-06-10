import { Badge } from "@/components/ui/badge/badge";
import Button from "@/components/ui/button/button";
import QueryState from "@/components/ui/query-state/query-state";
import type { GroupDetail } from "@/types/contracts";
import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  buildClassSelectorGroups,
  type ClassSelectorGroup,
  getClassSelectorAllIds,
} from "../../lib/class-selector";
import s from "./class-selector.module.scss";

type Props = {
  group: GroupDetail;
  value: string[];
  onChange: (next: string[]) => void;
  onAllIdsChange?: (ids: string[]) => void;
  initializeWithAll?: boolean;
  disabled?: boolean;
};

export const ClassSelector = ({
  group,
  value,
  onChange,
  onAllIdsChange,
  initializeWithAll = true,
  disabled = false,
}: Props) => {
  const groups = useMemo(() => (group ? buildClassSelectorGroups(group) : []), [group]);
  const allIds = useMemo(() => (group ? getClassSelectorAllIds(group) : []), [group]);
  const allIdsSet = useMemo(() => new Set(allIds), [allIds]);
  const selectedSet = useMemo(() => new Set(value), [value]);

  const groupIds = useMemo(() => groups.map((groupItem) => groupItem.id), [groups]);
  const groupIdsKey = useMemo(() => groupIds.join("|"), [groupIds]);

  const [openGroupIds, setOpenGroupIds] = useState<string[]>(groupIds);
  const initializedGroupIdRef = useRef<string | null>(null);

  useEffect(() => {
    setOpenGroupIds(groupIds);
  }, [groupIdsKey, groupIds]);

  useEffect(() => {
    onAllIdsChange?.(allIds);
  }, [allIds, onAllIdsChange]);

  useEffect(() => {
    if (!initializeWithAll) return;

    const hasForeignSelection = value.some((id) => !allIdsSet.has(id));
    const groupChanged = initializedGroupIdRef.current !== group.id;

    if (!groupChanged && !hasForeignSelection) return;

    onChange(allIds);
    initializedGroupIdRef.current = group.id;
  }, [allIds, allIdsSet, group.id, initializeWithAll, onChange, value]);

  const allSelected = allIds.length > 0 && allIds.every((id) => selectedSet.has(id));

  const handleToggleAll = () => {
    if (disabled) return;
    onChange(allSelected ? [] : allIds);
  };

  const handleToggleGroup = (groupItem: ClassSelectorGroup) => {
    if (disabled) return;

    const groupItemIds = groupItem.items.map((item) => item.id);
    const isGroupFullySelected = groupItemIds.every((id) => selectedSet.has(id));

    const next = isGroupFullySelected
      ? value.filter((id) => !groupItemIds.includes(id))
      : Array.from(new Set([...value, ...groupItemIds]));

    onChange(next);
  };

  const handleToggleItem = (itemId: string) => {
    if (disabled) return;

    const next = selectedSet.has(itemId) ? value.filter((id) => id !== itemId) : [...value, itemId];

    onChange(next);
  };

  const handleToggleOpen = (groupId: string) => {
    setOpenGroupIds((prev) =>
      prev.includes(groupId) ? prev.filter((id) => id !== groupId) : [...prev, groupId]
    );
  };

  const getGroupState = (groupItem: ClassSelectorGroup) => {
    const ids = groupItem.items.map((item) => item.id);
    const selectedCount = ids.filter((id) => selectedSet.has(id)).length;

    return {
      checked: ids.length > 0 && selectedCount === ids.length,
      indeterminate: selectedCount > 0 && selectedCount < ids.length,
      selectedCount,
      totalCount: ids.length,
    };
  };

  return (
    <div className={s.root}>
      <div className={s.header}>
        <div className={s.headerText}>
          <div className={s.eyebrow}>Настройка проверки</div>
          <div className={s.title}>Элементы контроля</div>
        </div>

        <div className={s.headerSide}>
          <Badge>{value.length}/{allIds.length}</Badge>
          <Button
            variant="ghost"
            size="sm"
            disabled={disabled || !allIds.length}
            onClick={handleToggleAll}
          >
            {allSelected ? "Снять все" : "Выбрать все"}
          </Button>
        </div>
      </div>

      <div className={s.body}>
        <QueryState
          isEmpty={!allIds.length}
          emptyTitle="Нет элементов"
          emptyDescription="Для выбранной группы не настроены элементы контроля"
        >
          <div className={s.groups}>
            {groups.map((groupItem) => {
              const groupState = getGroupState(groupItem);
              const isOpen = openGroupIds.includes(groupItem.id);
              const checkboxId = `inspection-group-${groupItem.id}`;

              return (
                <div key={groupItem.id} className={s.group}>
                  <div className={s.groupHead}>
                    <div className={s.groupMain}>
                      <input
                        id={checkboxId}
                        className={s.checkbox}
                        type="checkbox"
                        checked={groupState.checked}
                        disabled={disabled}
                        ref={(node) => {
                          if (node) {
                            node.indeterminate = groupState.indeterminate;
                          }
                        }}
                        onChange={() => handleToggleGroup(groupItem)}
                      />

                      <label
                        htmlFor={checkboxId}
                        className={clsx(s.groupLabel, disabled && s.interactiveDisabled)}
                      >
                        <span className={s.groupName}>{groupItem.name}</span>
                      </label>
                    </div>

                    <div className={s.groupMeta}>
                      <button
                        type="button"
                        className={s.groupToggle}
                        aria-expanded={isOpen}
                        onClick={() => handleToggleOpen(groupItem.id)}
                      >
                        {isOpen ? "Свернуть" : "Раскрыть"}
                      </button>

                      <Badge>
                        {groupState.selectedCount}/{groupState.totalCount}
                      </Badge>
                    </div>
                  </div>

                  {isOpen && (
                    <div className={s.groupBranch}>
                      {groupItem.items.map((item) => (
                        <label
                          key={item.id}
                          className={clsx(s.groupLeaf, disabled && s.interactiveDisabled)}
                        >
                          <input
                            className={s.checkbox}
                            type="checkbox"
                            checked={selectedSet.has(item.id)}
                            disabled={disabled}
                            onChange={() => handleToggleItem(item.id)}
                          />

                          <span
                            className={clsx(s.colorDot, item.hue == null && s.colorDotMuted)}
                            style={
                              item.hue == null
                                ? undefined
                                : { background: `hsl(${item.hue}, 70%, 50%)` }
                            }
                          />

                          <span className={s.itemName}>{item.name}</span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </QueryState>
      </div>
    </div>
  );
};
