import QueryState from "@/components/ui/query-state/query-state";
import { useGetInspection } from "@/page-components/inspections/api/get-inspection";
import { InspectionHistoryDetail } from "@/page-components/inspections/components/inspection-history-detail/inspection-history-detail";
import { inspectionModeLabel, inspectionStatusLabel } from "@/utils/labels";
import { useLoaderData, useNavigate } from "react-router-dom";
import { useInspectionHistoryOutletContext } from "./_inspection-history-group";
import p from "../platform-pages.module.scss";

export function Component() {
  const { inspectionId } = useLoaderData() as { inspectionId: string | null };
  const navigate = useNavigate();
  const { history, selectedGroup, buildInspectionPath } = useInspectionHistoryOutletContext();

  return (
    <QueryState isEmpty={!history.length} size="page" emptyTitle="No runs" emptyDescription={`Для «${selectedGroup.name}» пока нет сохранённых проверок.`}>
      <div className={p.grid2}>
        <section className={p.panelCard}>
          <div className={p.tableLike}>
            <div className={p.tableHead}><span>Reference</span><span>Mode</span><span>Status</span><span>Camera</span><span>Updated</span></div>
            {history.map((item) => <button type="button" className={p.tableRow} key={item.id} onClick={() => navigate(item.id === inspectionId ? buildInspectionPath(null) : buildInspectionPath(item.id))}><span>{item.standard_name || "—"}</span><span>{inspectionModeLabel(item.mode)}</span><span><b className={item.status === "passed" ? p.ok : undefined}>{inspectionStatusLabel(item.status)}</b></span><span>{item.camera_name || "—"}</span><span>{new Date(item.inspected_at).toLocaleString()}</span></button>)}
          </div>
        </section>
        <aside className={p.sidePanel}>{inspectionId ? <ExpandedInspectionHistoryDetail inspectionId={inspectionId} /> : <><h3>Run details</h3><p>Выбери проверку слева, чтобы открыть результат.</p></>}</aside>
      </div>
    </QueryState>
  );
}

function ExpandedInspectionHistoryDetail({ inspectionId }: { inspectionId: string }) {
  const { data } = useGetInspection(inspectionId);
  return <InspectionHistoryDetail inspection={data} />;
}
