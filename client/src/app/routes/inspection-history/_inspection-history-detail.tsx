import QueryState from "@/components/ui/query-state/query-state";
import { useGetInspection } from "@/page-components/inspections/api/get-inspection";
import { InspectionHistoryDetail } from "@/page-components/inspections/components/inspection-history-detail/inspection-history-detail";
import type { InspectionHistoryItem } from "@/types/contracts";
import { inspectionModeLabel, inspectionStatusLabel } from "@/utils/labels";
import clsx from "clsx";
import { AlertTriangle, Camera, CheckCircle2, Clock3, Image, ListChecks, ScanSearch } from "lucide-react";
import { useLoaderData, useNavigate } from "react-router-dom";
import { useInspectionHistoryOutletContext } from "./_inspection-history-group";
import p from "../platform-pages.module.scss";

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
    <QueryState isEmpty={!history.length} size="page" emptyTitle="No inspection runs" emptyDescription={`Для «${selectedGroup.name}» пока нет сохранённых проверок.`}>
      <div className={p.runsWorkbench}>
        <section className={p.runsMasterPanel}>
          <div className={p.cardTitleRow}>
            <div>
              <h3>Inspection timeline</h3>
              <p>Выбери запуск, чтобы открыть полный отчёт с изображениями и компонентами.</p>
            </div>
            <span className={p.softBadge}>{history.length}</span>
          </div>

          <div className={p.runCardList}>
            {history.map((item) => {
              const selected = item.id === inspectionId;
              const health = getRunHealth(item);
              const percent = getMatchedPercent(item);

              return (
                <button
                  type="button"
                  className={clsx(p.runTimelineCard, selected && p.runTimelineCardActive, p[`runTimeline_${health}`])}
                  key={item.id}
                  onClick={() => navigate(selected ? buildInspectionPath(null) : buildInspectionPath(item.id))}
                >
                  <div className={p.runTimelineThumb}>
                    {item.result_image_path || item.image_path ? <img src={`/storage/${item.result_image_path || item.image_path}`} alt="" /> : <Image />}
                    <span>{inspectionModeLabel(item.mode)}</span>
                  </div>
                  <div className={p.runTimelineBody}>
                    <div>
                      <strong>{item.standard_name || "Без эталона"}</strong>
                      <span>{item.camera_name || item.model_name || "Manual photo"}</span>
                    </div>
                    <div className={p.runTimelineMeta}>
                      <span><Clock3 /> {new Date(item.inspected_at).toLocaleString()}</span>
                      <span><ListChecks /> {item.matched_segments}/{item.total_segments}</span>
                    </div>
                    <div className={p.runTimelineProgress}><span style={{ width: `${percent}%` }} /></div>
                  </div>
                  <b>{inspectionStatusLabel(item.status)}</b>
                </button>
              );
            })}
          </div>
        </section>

        <section className={clsx(p.runsDetailPanel, !selectedRun && p.runsDetailPanelEmpty)}>
          {selectedRun && inspectionId ? (
            <>
              <header className={p.runDetailHero}>
                <div className={p.runDetailIcon}>{selectedRun.status === "passed" ? <CheckCircle2 /> : <AlertTriangle />}</div>
                <div>
                  <span className={p.eyebrow}><ScanSearch /> Inspection result</span>
                  <h3>{selectedRun.standard_name || "Inspection run"}</h3>
                  <p>{inspectionModeLabel(selectedRun.mode)} · {new Date(selectedRun.inspected_at).toLocaleString()} · {selectedRun.camera_name || "manual upload"}</p>
                </div>
                <div className={p.runDetailStats}>
                  <small><b>{selectedRun.matched_segments}</b> matched</small>
                  <small><b>{Math.max(0, selectedRun.total_segments - selectedRun.matched_segments)}</b> issues</small>
                  <small><b>{selectedRun.total_segments}</b> total</small>
                </div>
              </header>
              <ExpandedInspectionHistoryDetail inspectionId={inspectionId} />
            </>
          ) : (
            <div className={p.emptyFocusCard}>
              <Camera />
              <strong>Open an inspection report</strong>
              <span>Отчёт покажет результат, исходное фото, эталон и список отсутствующих/лишних компонентов.</span>
            </div>
          )}
        </section>
      </div>
    </QueryState>
  );
}

function ExpandedInspectionHistoryDetail({ inspectionId }: { inspectionId: string }) {
  const { data } = useGetInspection(inspectionId);
  return <InspectionHistoryDetail inspection={data} />;
}
