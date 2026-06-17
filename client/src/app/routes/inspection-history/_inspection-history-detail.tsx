
import QueryState from "@/components/ui/query-state/query-state";
import { useGetInspection } from "@/page-components/inspections/api/get-inspection";
import { InspectionHistoryDetail } from "@/page-components/inspections/components/inspection-history-detail/inspection-history-detail";
import type { InspectionHistoryItem } from "@/types/contracts";
import { inspectionModeLabel, inspectionStatusLabel } from "@/utils/labels";
import clsx from "clsx";
import { AlertTriangle, Camera, CheckCircle2, Clock3, Image, ListChecks, MousePointer2, ScanSearch, XCircle } from "lucide-react";
import { useLoaderData, useNavigate } from "react-router-dom";
import { formatInspectionHistoryDateTime } from "@/page-components/inspections/lib/inspection-history";
import { useInspectionHistoryOutletContext } from "./_inspection-history-group";
import s from "./_inspection-history-strict.module.scss";

function getRunHealth(item: InspectionHistoryItem) {
  if (item.status === "passed") return "passed";
  if (item.matched_segments === 0) return "failed";
  return "review";
}

function getMatchedPercent(item: InspectionHistoryItem) {
  if (!item.total_segments) return 0;
  return Math.max(0, Math.min(100, Math.round((item.matched_segments / item.total_segments) * 100)));
}

export function Component() {
  const { inspectionId } = useLoaderData() as { inspectionId: string | null };
  const navigate = useNavigate();
  const { history, selectedGroup, buildInspectionPath } = useInspectionHistoryOutletContext();
  const selectedRun = history.find((item) => item.id === inspectionId) ?? null;

  return (
    <QueryState
      isEmpty={!history.length}
      size="page"
      emptyTitle="Нет сохранённых проверок"
      emptyDescription={`Для «${selectedGroup.name}» пока нет отчётов. Запусти Inspect и сохрани результат.`}
    >
      <div className={s.historyWorkbench}>
        <section className={s.timelinePanel}>
          <div className={s.panelHead}>
            <div>
              <h2>Timeline</h2>
              <p>Список проверок с быстрым статусом и процентом совпадения.</p>
            </div>
            <span className={s.countBadge}>{history.length}</span>
          </div>

          <div className={s.runList}>
            {history.map((item) => (
              <RunTimelineCard
                key={item.id}
                item={item}
                selected={item.id === inspectionId}
                onClick={() => navigate(item.id === inspectionId ? buildInspectionPath(null) : buildInspectionPath(item.id))}
              />
            ))}
          </div>
        </section>

        <section className={clsx(s.reportPanel, !selectedRun && s.reportPanelEmpty)}>
          {selectedRun && inspectionId ? (
            <>
              <ReportHero item={selectedRun} />
              <ExpandedInspectionHistoryDetail inspectionId={inspectionId} />
            </>
          ) : (
            <div className={s.emptyReport}>
              <MousePointer2 />
              <strong>Выбери отчёт</strong>
              <span>Справа появится итог проверки, изображение результата, исходное фото и детализация по компонентам.</span>
            </div>
          )}
        </section>
      </div>
    </QueryState>
  );
}

function RunTimelineCard({ item, selected, onClick }: { item: InspectionHistoryItem; selected: boolean; onClick: () => void }) {
  const health = getRunHealth(item);
  const percent = getMatchedPercent(item);
  const issueCount = Math.max(0, item.total_segments - item.matched_segments);

  return (
    <button
      type="button"
      className={clsx(s.runCard, selected && s.runCardActive, s[`runCard_${health}`])}
      onClick={onClick}
    >
      <div className={s.runThumb}>
        {item.result_image_path || item.image_path ? <img src={`/storage/${item.result_image_path || item.image_path}`} alt="" /> : <Image />}
        <span>{inspectionModeLabel(item.mode)}</span>
      </div>

      <div className={s.runBody}>
        <div className={s.runTopline}>
          <strong>{item.standard_name || "Без эталона"}</strong>
          <span className={clsx(s.statusPill, health === "passed" && s.statusOk, health !== "passed" && s.statusWarn)}>{inspectionStatusLabel(item.status)}</span>
        </div>
        <p>{item.camera_name || item.model_name || "Manual photo"}</p>
        <div className={s.runMeta}>
          <span><Clock3 /> {formatInspectionHistoryDateTime(item.inspected_at)}</span>
          <span><ListChecks /> {item.matched_segments}/{item.total_segments}</span>
          <span>{issueCount ? <AlertTriangle /> : <CheckCircle2 />} {issueCount} issues</span>
        </div>
        <div className={s.runProgress} aria-label={`${percent}% matched`}>
          <span style={{ width: `${percent}%` }} />
        </div>
      </div>
    </button>
  );
}

function ReportHero({ item }: { item: InspectionHistoryItem }) {
  const health = getRunHealth(item);
  const issueCount = Math.max(0, item.total_segments - item.matched_segments);

  return (
    <header className={clsx(s.reportHero, health === "passed" && s.reportHeroPassed, health !== "passed" && s.reportHeroReview)}>
      <div className={s.reportIcon}>{health === "passed" ? <CheckCircle2 /> : health === "failed" ? <XCircle /> : <AlertTriangle />}</div>
      <div className={s.reportTitleBlock}>
        <span className={s.eyebrow}><ScanSearch /> Inspection result</span>
        <h2>{item.standard_name || "Inspection run"}</h2>
        <p>{inspectionModeLabel(item.mode)} · {formatInspectionHistoryDateTime(item.inspected_at)} · {item.camera_name || "manual upload"}</p>
      </div>
      <div className={s.reportStats}>
        <small><b>{item.matched_segments}</b> matched</small>
        <small><b>{issueCount}</b> issues</small>
        <small><b>{item.total_segments}</b> total</small>
      </div>
    </header>
  );
}

function ExpandedInspectionHistoryDetail({ inspectionId }: { inspectionId: string }) {
  const { data } = useGetInspection(inspectionId);
  return <InspectionHistoryDetail inspection={data} />;
}
