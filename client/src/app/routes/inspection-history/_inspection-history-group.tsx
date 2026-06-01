import { ContentHeader } from "@/components/layouts/content-header/content-header";
import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import QueryState from "@/components/ui/query-state/query-state";
import { useSearch } from "@/hooks/use-search";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { useGetInspectionHistory } from "@/page-components/inspections/api/get-inspection-history";
import { formatInspectionHistoryDateTime } from "@/page-components/inspections/lib/inspection-history";
import type { GroupListItem, InspectionHistoryItem } from "@/types/contracts";
import { inspectionModeLabel, inspectionStatusLabel } from "@/utils/labels";
import { X } from "lucide-react";
import { useCallback, useMemo } from "react";
import { Outlet, useLoaderData, useOutletContext } from "react-router-dom";
import { paths } from "../../paths";
import s from "./_inspection-history-group.module.scss";

type InspectionHistoryOutletContext = {
  groupId: string;
  selectedGroup: GroupListItem;
  history: InspectionHistoryItem[];
  buildInspectionPath: (inspectionId: string | null) => string;
};

export const useInspectionHistoryOutletContext = () => {
  return useOutletContext<InspectionHistoryOutletContext>();
};

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };

  const { data: groups } = useGetGroups();
  const { data: groupHistory } = useGetInspectionHistory(groupId);

  const groupsById = useMemo(() => buildGroupsById(groups), [groups]);
  const selectedGroup = groupsById[groupId] ?? null;

  const getInspectionSearchText = useCallback(
    (item: InspectionHistoryItem) => {
      const groupName = groupsById[item.group_id ?? ""]?.name ?? "";

      return [
        item.id,
        groupName,
        item.standard_name,
        item.camera_name,
        item.model_name,
        item.mode,
        inspectionModeLabel(item.mode),
        item.status,
        inspectionStatusLabel(item.status),
        item.notes,
        formatInspectionHistoryDateTime(item.inspected_at),
      ]
        .filter(Boolean)
        .join(" ");
    },
    [groupsById]
  );

  const {
    search,
    setSearch,
    filtered: filteredHistory,
    resetSearch,
  } = useSearch({
    items: groupHistory,
    getText: getInspectionSearchText,
  });

  if (!selectedGroup) {
    return (
      <QueryState
        isEmpty
        size="page"
        emptyTitle="Группа не найдена"
        emptyDescription="Выберите другую группу в списке слева."
      />
    );
  }

  const passedCount = groupHistory.filter((item) => item.status === "passed").length;
  const failedCount = groupHistory.filter((item) => item.status === "failed").length;

  const hasSearch = Boolean(search.trim());
  const historyCountLabel = hasSearch
    ? `${filteredHistory.length} из ${groupHistory.length}`
    : String(groupHistory.length);

  return (
    <>
      <ContentHeader>
        <ContentHeader.Top
          title={selectedGroup.name}
          subtitles={["История контроля выбранной группы"]}
          meta={[
            `${historyCountLabel} проверок`,
            `${passedCount} пройдено`,
            `${failedCount} не пройдено`,
          ]}
        >
          <ContentHeader.Actions>
            <></>
          </ContentHeader.Actions>
        </ContentHeader.Top>

        <div className={s.controls}>
          <Input
            noMargin
            placeholder="Поиск по эталону, камере, модели..."
            value={search}
            onChange={setSearch}
          />
          <Button icon={X} variant="ghost" disabled={!hasSearch} onClick={resetSearch}>
            Сбросить
          </Button>
        </div>
      </ContentHeader>

      <Outlet
        context={{
          groupId,
          selectedGroup,
          history: filteredHistory,
          buildInspectionPath: (inspectionId: string | null) =>
            inspectionId
              ? paths.inspectionHistoryDetail(groupId, inspectionId)
              : paths.inspectionHistoryGroup(groupId),
        }}
      />
    </>
  );
}

function buildGroupsById(groups: GroupListItem[]): Record<string, GroupListItem> {
  return Object.fromEntries(groups.map((group) => [group.id, group]));
}
