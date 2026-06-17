import { paths, type InspectionModePath } from "@/app/paths";
import Button from "@/components/ui/button/button";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { getCamerasQueryOptions } from "@/page-components/cameras/api/get-cameras";
import { InspectionResultPanel } from "@/page-components/inspections/components/inspection-result-panel/inspection-result-panel";
import { useInspectionLayout } from "@/page-components/inspections/hooks/use-inspection-layout";
import type { SegmentClass } from "@/types/contracts";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  CircleDot,
  Filter,
  Image as ImageIcon,
  ListChecks,
  RadioTower,
  Upload,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { Outlet, useLoaderData, useNavigate, useParams } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { mode: currentMode } = useLoaderData() as { mode: InspectionModePath };
  const { groupId, standardId } = useParams();
  const navigate = useNavigate();
  const inspection = useInspectionLayout({ currentMode, groupId: groupId ?? null, standardId: standardId ?? null });
  const { data: groups = [] } = useGetGroups();
  const { data: cameras = [] } = useQuery(getCamerasQueryOptions());

  const standards = inspection.group?.standards ?? [];
  const classSource = inspection.standard ?? inspection.group;
  const classes = getClasses(classSource);
  const selectedResult = currentMode === "realtime" && inspection.realtimeSessionId ? inspection.realtimeStatus : inspection.result;
  const resultDetails = selectedResult?.details ?? [];
  const issueDetails = resultDetails.filter((detail) => detail.status !== "ok");
  const okDetails = resultDetails.filter((detail) => detail.status === "ok");
  const statusText = selectedResult?.status ?? "idle";
  const selectedClassesCount = inspection.selectedClassIds.length || classes.length;

  const modeCards = [
    { mode: "photo", icon: Upload, label: "Photo", hint: "manual upload" },
    { mode: "snapshot", icon: Camera, label: "Snapshot", hint: "single camera frame" },
    { mode: "realtime", icon: RadioTower, label: "Realtime", hint: "continuous control" },
  ] as const;

  const changeMode = (nextMode: InspectionModePath) => {
    if (groupId && standardId) return navigate(paths.inspectionStandard(nextMode, groupId, standardId));
    if (groupId) return navigate(paths.inspectionGroup(nextMode, groupId));
    navigate(paths.inspectionMode(nextMode));
  };

  const readiness = {
    dataset: Boolean(groupId),
    reference: Boolean(standardId),
    source: currentMode === "photo" ? Boolean(standardId) : Boolean(inspection.cameraId),
    result: Boolean(selectedResult),
  };

  return (
    <div className={`${p.page} ${p.inspectStationPageV17}`}>
      <header className={p.inspectHeroV17}>
        <div className={p.inspectHeroMainV17}>
          <span className={p.eyebrow}><Zap /> Inspect station</span>
          <h1>Проверка изделия</h1>
          <p>Выбери dataset, эталонный вид и источник изображения. Результат должен читаться как операторский отчёт, а не как техническая форма.</p>
        </div>
        <div className={p.inspectModeSwitchV17} aria-label="Inspection mode">
          {modeCards.map(({ mode, icon: Icon, label, hint }) => (
            <button type="button" key={mode} className={currentMode === mode ? p.modeActiveV17 : undefined} onClick={() => changeMode(mode)}>
              <Icon />
              <span>{label}</span>
              <small>{hint}</small>
            </button>
          ))}
        </div>
      </header>

      <section className={p.inspectSetupCardV17}>
        <div className={p.inspectSelectGridV17}>
          <ControlSelect
            label="Dataset"
            value={groupId ?? ""}
            onChange={(value) => navigate(value ? paths.inspectionGroup(currentMode, value) : paths.inspectionMode(currentMode))}
            placeholder="Select dataset"
            options={groups.map((group) => ({ value: group.id, label: group.name }))}
          />
          <ControlSelect
            label="Reference"
            value={standardId ?? ""}
            disabled={!groupId}
            onChange={(value) => groupId && navigate(value ? paths.inspectionStandard(currentMode, groupId, value) : paths.inspectionGroup(currentMode, groupId))}
            placeholder="Select reference"
            options={standards.map((standard) => ({ value: standard.id, label: standard.name }))}
          />
          <ControlSelect
            label="Camera"
            value={inspection.cameraId ?? ""}
            disabled={currentMode === "photo"}
            onChange={(value) => inspection.setCameraId(value || null)}
            placeholder={currentMode === "photo" ? "Not needed for photo" : "Select camera"}
            options={cameras.map((camera) => ({ value: camera.id, label: camera.name }))}
          />
          <div className={p.inspectClassSummaryV17}>
            <span><Filter /> Classes</span>
            <b>{selectedClassesCount || 0}</b>
            <small>{inspection.selectedClassIds.length ? "custom filter" : "all selected"}</small>
          </div>
        </div>
      </section>

      <section className={p.inspectPipelineV17} aria-label="Inspection setup progress">
        <PipelineStep done={readiness.dataset} active={!readiness.dataset} index="1" icon={ImageIcon} title="Dataset" text={inspection.group?.name ?? "Choose изделие"} />
        <PipelineStep done={readiness.reference} active={readiness.dataset && !readiness.reference} muted={!readiness.dataset} index="2" icon={CircleDot} title="Reference" text={inspection.standard?.name ?? "Select эталон"} />
        <PipelineStep done={readiness.source} active={readiness.reference && !readiness.source} muted={!readiness.reference} index="3" icon={currentMode === "photo" ? Upload : Camera} title="Source" text={currentMode === "photo" ? "Upload photo" : inspection.cameraId ? "Camera ready" : "Choose camera"} />
        <PipelineStep done={readiness.result} active={readiness.source && !readiness.result} muted={!readiness.source} index="4" icon={ListChecks} title="Result" text={selectedResult ? `${okDetails.length}/${resultDetails.length} matched` : "Run inspection"} />
      </section>

      <div className={p.inspectStationGridV17}>
        <section className={p.inspectConsoleV17}>
          <div className={p.inspectConsoleToolbarV17}>
            <div>
              <strong>{inspection.standard?.name ?? "Inspection canvas"}</strong>
              <p>{inspection.group?.name ?? "Dataset не выбран"} · {currentMode}</p>
            </div>
            <div className={p.inspectRunClusterV17}>
              <span className={selectedResult ? p.statusReadyV17 : p.statusIdleV17}>{statusText}</span>
              <Button disabled={inspection.outletContext.runDisabled} onClick={inspection.outletContext.onRun}>{inspection.outletContext.runLabel}</Button>
            </div>
          </div>
          <div className={p.inspectStageShellV17}>
            <Outlet context={inspection.outletContext} />
          </div>
        </section>

        <aside className={p.inspectInspectorV17}>
          <div className={p.inspectorHeaderV17}>
            <div>
              <span>Inspector</span>
              <h3>{issueDetails.length ? "Needs review" : selectedResult ? "Passed view" : "Ready for run"}</h3>
            </div>
            <b>{issueDetails.length}</b>
          </div>

          <div className={p.inspectorMetricGridV17}>
            <Metric label="Classes" value={classes.length} />
            <Metric label="Checked" value={resultDetails.length} />
            <Metric label="Matched" value={okDetails.length} tone="ok" />
            <Metric label="Issues" value={issueDetails.length} tone={issueDetails.length ? "bad" : "ok"} />
          </div>

          <div className={p.inspectorSectionV17}>
            <div className={p.inspectorSectionHeadV17}><strong>Class filter</strong><small>{inspection.selectedClassIds.length || "all"}</small></div>
            <div className={p.classListV17}>
              {classes.map((item) => {
                const checked = inspection.selectedClassIds.includes(item.id);
                return (
                  <label className={checked ? `${p.checkRowV17} ${p.checkRowActiveV17}` : p.checkRowV17} key={item.id}>
                    <input type="checkbox" checked={checked} onChange={(event) => {
                      inspection.setSelectedClassIds(event.target.checked ? [...inspection.selectedClassIds, item.id] : inspection.selectedClassIds.filter((id) => id !== item.id));
                    }} />
                    <span className={p.checkDotV17} style={{ backgroundColor: `hsl(${item.hue}, 78%, 52%)` }} />
                    <span>{item.name}</span>
                  </label>
                );
              })}
              {!classes.length ? <div className={p.emptyMiniV17}>Классы появятся после настройки dataset.</div> : null}
            </div>
          </div>

          <div className={p.inspectorSectionV17}>
            <div className={p.inspectorSectionHeadV17}><strong>Result details</strong><small>{statusText}</small></div>
            <div className={p.resultListV17}>
              {resultDetails.slice(0, 18).map((detail) => (
                <div className={p.resultItemV17} key={detail.annotation_id ?? detail.class_key}>
                  <span className={detail.status === "ok" ? p.resultOkDotV17 : detail.status === "missing" ? p.resultBadDotV17 : p.resultWarnDotV17} />
                  <span>{detail.name}</span>
                  <b>{detail.status}</b>
                </div>
              ))}
              {!selectedResult ? <div className={p.emptyMiniV17}>После запуска здесь появятся совпадения, отсутствующие и лишние детали.</div> : null}
            </div>
          </div>

          <div className={p.resultPanelWrapV17}>
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
          </div>
        </aside>
      </div>
    </div>
  );
}

function ControlSelect({ label, value, options, placeholder, disabled, onChange }: { label: string; value: string; placeholder: string; disabled?: boolean; options: { value: string; label: string }[]; onChange: (value: string) => void }) {
  return (
    <label className={p.controlSelectV17}>
      <span>{label}</span>
      <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
        <option value="">{placeholder}</option>
        {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>
  );
}

function PipelineStep({ index, icon: Icon, title, text, done, active, muted }: { index: string; icon: LucideIcon; title: string; text: string; done?: boolean; active?: boolean; muted?: boolean }) {
  const className = [p.pipelineStepV17, done ? p.pipelineDoneV17 : "", active ? p.pipelineActiveV17 : "", muted ? p.pipelineMutedV17 : ""].filter(Boolean).join(" ");
  return (
    <div className={className}>
      <span>{done ? <CheckCircle2 /> : index}</span>
      <Icon />
      <div><b>{title}</b><small>{text}</small></div>
    </div>
  );
}

function Metric({ label, value, tone }: { label: string; value: number; tone?: "ok" | "bad" }) {
  const className = [p.inspectorMetricV17, tone === "ok" ? p.metricOkV17 : "", tone === "bad" ? p.metricBadV17 : ""].filter(Boolean).join(" ");
  return <div className={className}><b>{value}</b><span>{label}</span></div>;
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
