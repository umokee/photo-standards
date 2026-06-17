
import { paths } from "@/app/paths";
import Input from "@/components/ui/input/input";
import QueryState from "@/components/ui/query-state/query-state";
import { useSearch } from "@/hooks/use-search";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { useGetInspectionHistory } from "@/page-components/inspections/api/get-inspection-history";
import { formatInspectionHistoryDateTime } from "@/page-components/inspections/lib/inspection-history";
import type { GroupListItem, InspectionHistoryItem } from "@/types/contracts";
import { inspectionModeLabel, inspectionStatusLabel } from "@/utils/labels";
import clsx from "clsx";
import { AlertTriangle, ArrowLeft, Camera, CheckCircle2, Clock3, History, Image, ListChecks, Search, type LucideIcon } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { Link, Outlet, useLoaderData, useOutletContext } from "react-router-dom";
import s from "./_inspection-history-strict.module.scss";

type StatusFilter = "all" | "passed" | "issues";
type ModeFilter = "all" | "photo" | "snapshot" | "realtime";

type InspectionHistoryOutletContext = {
  groupId: string;
  selectedGroup: GroupListItem;
  history: InspectionHistoryItem[];
  buildInspectionPath: (inspectionId: string | null) => string;
};

export const useInspectionHistoryOutletContext = () => useOutletContext<InspectionHistoryOutletContext>();

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: groups } = useGetGroups();
  const { data: groupHistory } = useGetInspectionHistory(groupId);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [modeFilter, setModeFilter] = useState<ModeFilter>("all");

  const groupsById = useMemo(() => Object.fromEntries(groups.map((group) => [group.id, group])), [groups]);
  const selectedGroup = groupsById[groupId] ?? null;

  const getInspectionSearchText = useCallback((item: InspectionHistoryItem) => {
    return [
      item.id,
      item.standard_name,
      item.camera_name,
      item.model_name,
      item.mode,
      inspectionModeLabel(item.mode),
      item.status,
      inspectionStatusLabel(item.status),
      item.notes,
      formatInspectionHistoryDateTime(item.inspected_at),
    ].filter(Boolean).join(" ");
  }, []);

  const { search, setSearch, filtered: searchedRuns } = useSearch({ items: groupHistory, getText: getInspectionSearchText });

  const filtered = useMemo(() => {
    return searchedRuns.filter((item) => {
      const statusMatch = statusFilter === "all" || (statusFilter === "passed" ? item.status === "passed" : item.status !== "passed");
      const modeMatch = modeFilter === "all" || item.mode === modeFilter;
      return statusMatch && modeMatch;
    });
  }, [modeFilter, searchedRuns, statusFilter]);

  const passedRuns = filtered.filter((item) => item.status === "passed").length;
  const issueRuns = filtered.filter((item) => item.status !== "passed").length;
  const photoRuns = filtered.filter((item) => item.mode === "photo").length;
  const snapshotRuns = filtered.filter((item) => item.mode === "snapshot").length;
  const realtimeRuns = filtered.filter((item) => item.mode === "realtime").length;
  const latestRun = filtered[0] ?? null;

  if (!selectedGroup) {
    return <QueryState isEmpty size="page" emptyTitle="Изделие не найдено" />;
  }

  return (
    <main className={s.page}>
      <section className={s.groupHeader}>
        <div className={s.groupTitleBlock}>
          <Link className={s.backLink} to={paths.inspectionHistory()}><ArrowLeft /> Runs</Link>
          <span className={s.eyebrow}><History /> Inspection reports</span>
          <h1>{selectedGroup.name}</h1>
          <p>{filtered.length} сохранённых проверок · {selectedGroup.stats.standards_count} эталонов · {selectedGroup.stats.segment_classes_count} классов</p>
        </div>
        <div className={s.groupHeaderActions}>
          <Link className={s.secondaryAction} to={paths.inspectionMode("photo")}><Image /> Проверить фото</Link>
          <Link className={s.primaryAction} to={paths.inspectionGroup("snapshot", selectedGroup.id)}><Camera /> Snapshot</Link>
        </div>
      </section>

      <section className={s.summaryGrid}>
        <SummaryCard value={filtered.length} label="Total" hint="reports" icon={ListChecks} />
        <SummaryCard value={passedRuns} label="Passed" hint="without issues" icon={CheckCircle2} tone="ok" />
        <SummaryCard value={issueRuns} label="Need review" hint="missing/extra" icon={AlertTriangle} tone="warn" />
        <SummaryCard value={photoRuns + snapshotRuns + realtimeRuns} label="Modes" hint={`${photoRuns}/${snapshotRuns}/${realtimeRuns}`} icon={Clock3} />
      </section>

      <section className={s.filterPanel}>
        <div className={s.searchShell}>
          <Search />
          <Input noMargin placeholder="Поиск по эталону, камере, модели, статусу..." value={search} onChange={setSearch} />
        </div>

        <div className={s.filterRow}>
          <SegmentButton active={statusFilter === "all"} onClick={() => setStatusFilter("all")}>Все</SegmentButton>
          <SegmentButton active={statusFilter === "passed"} onClick={() => setStatusFilter("passed")}>Passed</SegmentButton>
          <SegmentButton active={statusFilter === "issues"} onClick={() => setStatusFilter("issues")}>Issues</SegmentButton>
          <span className={s.filterDivider} />
          <SegmentButton active={modeFilter === "all"} onClick={() => setModeFilter("all")}>All modes</SegmentButton>
          <SegmentButton active={modeFilter === "photo"} onClick={() => setModeFilter("photo")}>Photo</SegmentButton>
          <SegmentButton active={modeFilter === "snapshot"} onClick={() => setModeFilter("snapshot")}>Snapshot</SegmentButton>
          <SegmentButton active={modeFilter === "realtime"} onClick={() => setModeFilter("realtime")}>Realtime</SegmentButton>
        </div>

        {latestRun ? (
          <div className={s.latestStrip}>
            <Clock3 /> Последний отчёт: {formatInspectionHistoryDateTime(latestRun.inspected_at)} · {latestRun.standard_name || "без эталона"}
          </div>
        ) : null}
      </section>

      <Outlet context={{
        groupId,
        selectedGroup,
        history: filtered,
        buildInspectionPath: (inspectionId: string | null) => inspectionId ? paths.inspectionHistoryDetail(groupId, inspectionId) : paths.inspectionHistoryGroup(groupId),
      }} />
    </main>
  );
}

function SummaryCard({ value, label, hint, icon: Icon, tone = "neutral" }: { value: number; label: string; hint: string; icon: LucideIcon; tone?: "neutral" | "ok" | "warn" }) {
  return (
    <div className={clsx(s.summaryCard, tone === "ok" && s.summaryOk, tone === "warn" && s.summaryWarn)}>
      <Icon />
      <b>{value}</b>
      <span>{label}</span>
      <small>{hint}</small>
    </div>
  );
}

function SegmentButton({ active, children, onClick }: { active: boolean; children: string; onClick: () => void }) {
  return <button type="button" className={clsx(s.segmentButton, active && s.segmentButtonActive)} onClick={onClick}>{children}</button>;
}
