import Input from "@/components/ui/input/input";
import QueryState from "@/components/ui/query-state/query-state";
import { useSearch } from "@/hooks/use-search";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { useGetInspectionHistory } from "@/page-components/inspections/api/get-inspection-history";
import { formatInspectionHistoryDateTime } from "@/page-components/inspections/lib/inspection-history";
import type { GroupListItem, InspectionHistoryItem } from "@/types/contracts";
import { inspectionModeLabel, inspectionStatusLabel } from "@/utils/labels";
import { useCallback, useMemo } from "react";
import { Outlet, useLoaderData, useOutletContext } from "react-router-dom";
import { paths } from "../../paths";
import p from "../platform-pages.module.scss";

type InspectionHistoryOutletContext = { groupId: string; selectedGroup: GroupListItem; history: InspectionHistoryItem[]; buildInspectionPath: (inspectionId: string | null) => string };
export const useInspectionHistoryOutletContext = () => useOutletContext<InspectionHistoryOutletContext>();

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: groups } = useGetGroups();
  const { data: groupHistory } = useGetInspectionHistory(groupId);
  const groupsById = useMemo(() => Object.fromEntries(groups.map((group) => [group.id, group])), [groups]);
  const selectedGroup = groupsById[groupId] ?? null;
  const getInspectionSearchText = useCallback((item: InspectionHistoryItem) => [item.id, item.standard_name, item.camera_name, item.model_name, item.mode, inspectionModeLabel(item.mode), item.status, inspectionStatusLabel(item.status), item.notes, formatInspectionHistoryDateTime(item.inspected_at)].filter(Boolean).join(" "), []);
  const { search, setSearch, filtered } = useSearch({ items: groupHistory, getText: getInspectionSearchText });

  if (!selectedGroup) return <QueryState isEmpty size="page" emptyTitle="Dataset not found" />;

  return (
    <div className={p.page}>
      <header className={p.ultraHeader}><div><h1>{selectedGroup.name}</h1><p>{filtered.length} inspection runs</p></div></header>
      <div className={p.panelCard}><Input noMargin placeholder="Search runs..." value={search} onChange={setSearch} /></div>
      <Outlet context={{ groupId, selectedGroup, history: filtered, buildInspectionPath: (inspectionId: string | null) => inspectionId ? paths.inspectionHistoryDetail(groupId, inspectionId) : paths.inspectionHistoryGroup(groupId) }} />
    </div>
  );
}
