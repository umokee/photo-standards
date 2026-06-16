import { paths, type InspectionModePath } from "@/app/paths";
import Button from "@/components/ui/button/button";
import { getCamerasQueryOptions } from "@/page-components/cameras/api/get-cameras";
import { InspectionResultPanel } from "@/page-components/inspections/components/inspection-result-panel/inspection-result-panel";
import { useInspectionLayout } from "@/page-components/inspections/hooks/use-inspection-layout";
import type { SegmentClass } from "@/types/contracts";
import { useQuery } from "@tanstack/react-query";
import { Camera, RadioTower, Upload } from "lucide-react";
import { Outlet, useLoaderData, useNavigate, useParams } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { mode: currentMode } = useLoaderData() as { mode: InspectionModePath };
  const { groupId, standardId } = useParams();
  const navigate = useNavigate();
  const inspection = useInspectionLayout({ currentMode, groupId: groupId ?? null, standardId: standardId ?? null });
  const { data: cameras = [] } = useQuery(getCamerasQueryOptions());

  const standards = inspection.group?.standards ?? [];
  const classSource = inspection.standard ?? inspection.group;
  const classes = getClasses(classSource);
  const selectedResult = currentMode === "realtime" && inspection.realtimeSessionId ? inspection.realtimeStatus : inspection.result;

  const modeCards = [
    { mode: "photo", icon: Upload, label: "Photo" },
    { mode: "snapshot", icon: Camera, label: "Snapshot" },
    { mode: "realtime", icon: RadioTower, label: "Realtime" },
  ] as const;

  const changeMode = (nextMode: InspectionModePath) => {
    if (groupId && standardId) return navigate(paths.inspectionStandard(nextMode, groupId, standardId));
    if (groupId) return navigate(paths.inspectionGroup(nextMode, groupId));
    navigate(paths.inspectionMode(nextMode));
  };

  return (
    <div className={p.page}>
      <header className={p.ultraHeader}>
        <div><h1>Deploy</h1><p>Проверка изделия по эталону: фото, snapshot с камеры или realtime-контроль.</p></div>
        <div className={p.tabs}>{modeCards.map(({ mode, icon: Icon, label }) => <button type="button" key={mode} className={currentMode === mode ? p.active : undefined} onClick={() => changeMode(mode)}><Icon />{label}</button>)}</div>
      </header>

      <section className={p.panelCard}>
        <div className={p.selectGrid}>
          <div className={p.field}>
            <label>Dataset</label>
            <select value={groupId ?? ""} onChange={(event) => navigate(event.target.value ? paths.inspectionGroup(currentMode, event.target.value) : paths.inspectionMode(currentMode))}>
              <option value="">Select dataset</option>
              {inspection.group ? <option value={inspection.group.id}>{inspection.group.name}</option> : null}
            </select>
          </div>
          <div className={p.field}>
            <label>Reference</label>
            <select value={standardId ?? ""} disabled={!groupId} onChange={(event) => groupId && navigate(event.target.value ? paths.inspectionStandard(currentMode, groupId, event.target.value) : paths.inspectionGroup(currentMode, groupId))}>
              <option value="">Select reference</option>
              {standards.map((standard) => <option key={standard.id} value={standard.id}>{standard.name}</option>)}
            </select>
          </div>
          <div className={p.field}>
            <label>Camera</label>
            <select value={inspection.cameraId ?? ""} disabled={currentMode === "photo"} onChange={(event) => inspection.setCameraId(event.target.value || null)}>
              <option value="">Select camera</option>
              {cameras.map((camera) => <option key={camera.id} value={camera.id}>{camera.name}</option>)}
            </select>
          </div>
          <div className={p.field}>
            <label>Classes</label>
            <select value={inspection.selectedClassIds.length ? "custom" : ""} onChange={() => {}} disabled={!classes.length}>
              <option>{inspection.selectedClassIds.length ? `${inspection.selectedClassIds.length} selected` : "Select below"}</option>
            </select>
          </div>
        </div>
      </section>

      <div className={p.inspectGrid}>
        <section className={p.inspectCard}>
          <div className={p.inspectToolbar}>
            <div><strong>{inspection.standard?.name ?? "Inspection canvas"}</strong><p>{inspection.group?.name ?? "Choose dataset and reference"}</p></div>
            <Button disabled={inspection.outletContext.runDisabled} onClick={inspection.outletContext.onRun}>{inspection.outletContext.runLabel}</Button>
          </div>
          <div className={p.inspectStage}>
            <Outlet context={inspection.outletContext} />
          </div>
        </section>

        <aside className={p.sidePanel}>
          <h3>Inspector</h3>
          <p>Фильтр классов и результат проверки.</p>
          <div className={p.classList}>
            {classes.map((item) => {
              const checked = inspection.selectedClassIds.includes(item.id);
              return (
                <label className={p.checkRow} key={item.id}>
                  <input type="checkbox" checked={checked} onChange={(event) => {
                    inspection.setSelectedClassIds(event.target.checked ? [...inspection.selectedClassIds, item.id] : inspection.selectedClassIds.filter((id) => id !== item.id));
                  }} />
                  <span className={p.checkDot} style={{ backgroundColor: `hsl(${item.hue}, 75%, 52%)` }} />
                  <span>{item.name}</span>
                </label>
              );
            })}
          </div>

          <div className={p.resultList}>
            {selectedResult?.details?.slice(0, 12).map((detail) => (
              <div className={p.resultItem} key={detail.annotation_id ?? detail.class_key}>
                <span className={detail.status === "ok" ? p.resultStatusOk : detail.status === "missing" ? p.resultStatusBad : p.resultStatusWarn} />
                <span>{detail.name}</span>
                <b>{detail.status}</b>
              </div>
            )) ?? null}
          </div>

          {currentMode === "realtime" && inspection.realtimeSessionId ? (
            <InspectionResultPanel
              kind="realtime"
              result={inspection.realtimeStatus}
              sessionId={inspection.realtimeSessionId}
              activeMatchKey={inspection.focus.activeMatchKey}
              setActiveMatchKey={inspection.focus.setActiveMatchKey}
            />
          ) : inspection.result ? (
            <InspectionResultPanel
              result={inspection.result}
              savedInspectionId={inspection.savedInspectionId}
              onSaved={inspection.setSavedInspectionId}
              onDiscarded={inspection.resetInspectionState}
              activeMatchKey={inspection.focus.activeMatchKey}
              setActiveMatchKey={inspection.focus.setActiveMatchKey}
            />
          ) : null}
        </aside>
      </div>
    </div>
  );
}

function getClasses(source: unknown): SegmentClass[] {
  if (!source || typeof source !== "object") return [];
  const value = source as {
    segment_class_categories?: { segment_classes: SegmentClass[] }[];
    ungrouped_segment_classes?: SegmentClass[];
  };
  return [
    ...(value.segment_class_categories?.flatMap((category) => category.segment_classes) ?? []),
    ...(value.ungrouped_segment_classes ?? []),
  ];
}
