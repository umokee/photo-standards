import { inspectionModePaths, paths, type InspectionModePath } from "@/app/paths";
import { SplitLayout } from "@/components/layouts/split-layout/split-layout";
import QueryState from "@/components/ui/query-state/query-state";
import { ClassSelector } from "@/page-components/inspections/components/class-selector/class-selector";
import { InspectionResultPanel } from "@/page-components/inspections/components/inspection-result-panel/inspection-result-panel";
import { InspectionSourceControl } from "@/page-components/inspections/components/inspection-source-control/inspection-source-control";
import { useInspectionLayout } from "@/page-components/inspections/hooks/use-inspection-layout";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  CircleDot,
  FileImage,
  Image,
  Layers3,
  ListChecks,
  MousePointer2,
  ShieldCheck,
} from "lucide-react";
import { Link, Outlet, useLoaderData, useNavigate, useParams } from "react-router-dom";
import p from "../platform-pages.module.scss";

const modeLabel: Record<InspectionModePath, string> = {
  photo: "Photo",
  snapshot: "Snapshot",
  realtime: "Realtime",
};

const modeSourceLabel: Record<InspectionModePath, string> = {
  photo: "Файл изображения",
  snapshot: "Снимок с камеры",
  realtime: "Live camera",
};

export function Component() {
  const { mode: currentMode } = useLoaderData() as { mode: InspectionModePath };
  const { groupId, standardId } = useParams();
  const navigate = useNavigate();
  const inspection = useInspectionLayout({
    currentMode,
    groupId: groupId ?? null,
    standardId: standardId ?? null,
  });

  const handleModeChange = (nextMode: InspectionModePath) => {
    if (groupId && standardId) {
      navigate(paths.inspectionStandard(nextMode, groupId, standardId));
      return;
    }

    if (groupId) {
      navigate(paths.inspectionGroup(nextMode, groupId));
      return;
    }

    navigate(paths.inspectionMode(nextMode));
  };

  const selectedClassesCount = inspection.selectedClassIds.length;
  const hasSource =
    currentMode === "photo"
      ? Boolean(inspection.file)
      : currentMode === "snapshot" || currentMode === "realtime"
        ? Boolean(inspection.cameraId)
        : false;
  const stationReady = Boolean(groupId && standardId && selectedClassesCount > 0 && hasSource);
  const isSelectionStage = !standardId;

  const renderPanel = () => {
    if (currentMode === "realtime" && inspection.realtimeSessionId) {
      return (
        <InspectionResultPanel
          kind="realtime"
          result={inspection.realtimeStatus}
          sessionId={inspection.realtimeSessionId}
          activeMatchKey={inspection.focus.activeMatchKey}
          setActiveMatchKey={inspection.focus.setActiveMatchKey}
        />
      );
    }

    if (!groupId) {
      return (
        <div className={p.inspectBlockedPanelV33}>
          <MousePointer2 />
          <h3>Сначала выбери изделие</h3>
          <p>На этом шаге справа не должно быть классов или эталонов: выбор проекта происходит в основной области.</p>
        </div>
      );
    }

    if (!standardId) {
      if (inspection.groupQuery.isPending) {
        return <QueryState isLoading size="block" loadingText="Загружаем изделие" />;
      }

      if (inspection.groupQuery.isError) {
        return (
          <QueryState
            isError
            size="block"
            errorTitle="Не удалось загрузить изделие"
            errorDescription="Проверь выбранный проект и попробуй снова."
          />
        );
      }

      if (!inspection.group) {
        return <QueryState isEmpty size="block" emptyTitle="Изделие не найдено" />;
      }

      return (
        <div className={p.inspectBlockedPanelV33}>
          <Layers3 />
          <h3>Теперь выбери reference</h3>
          <p>
            Классы и запуск проверки появятся после выбора эталонного вида. Так station не дублирует
            выбор project/reference в верхней панели.
          </p>
          <div className={p.inspectBlockedStatsV33}>
            <span><Image /> {inspection.group.standards.length} references</span>
            <span><ShieldCheck /> {inspection.group.stats.segment_classes_count} classes</span>
            <span><ListChecks /> {inspection.group.stats.polygons_count} polygons</span>
          </div>
          <Link to={paths.assetReferences(inspection.group.id)}>Открыть references в Assets</Link>
        </div>
      );
    }

    if (inspection.result) {
      return (
        <InspectionResultPanel
          result={inspection.result}
          savedInspectionId={inspection.savedInspectionId}
          onSaved={inspection.setSavedInspectionId}
          onDiscarded={inspection.resetInspectionState}
          activeMatchKey={inspection.focus.activeMatchKey}
          setActiveMatchKey={inspection.focus.setActiveMatchKey}
        />
      );
    }

    if (inspection.groupQuery.isPending || inspection.standardQuery.isPending) {
      return <QueryState isLoading size="block" loadingText="Подготавливаем station" />;
    }

    if (inspection.groupQuery.isError || inspection.standardQuery.isError) {
      return (
        <QueryState
          isError
          size="block"
          errorTitle="Не удалось подготовить station"
          errorDescription="Проверь выбранные изделие и reference."
        />
      );
    }

    if (!inspection.group || !inspection.standard) {
      return <QueryState isEmpty size="block" emptyTitle="Station не готов" />;
    }

    return (
      <div className={p.inspectPanelInnerV32}>
        <div className={p.inspectPanelHeadV32}>
          <span>Classes</span>
          <small>Выбери детали, которые нужно проверить в этом запуске.</small>
        </div>
        <ClassSelector
          source={inspection.standard}
          value={inspection.selectedClassIds}
          onChange={inspection.setSelectedClassIds}
          disabled={inspection.isLocked}
        />
      </div>
    );
  };

  return (
    <SplitLayout>
      <SplitLayout.Content>
        <SplitLayout.Topbar>
          {isSelectionStage ? (
            <SelectionTopbar
              currentMode={currentMode}
              groupName={inspection.group?.name ?? null}
              isLocked={inspection.isLocked}
              taskStage={inspection.taskStage ?? null}
              onModeChange={handleModeChange}
            />
          ) : (
            <StationTopbar
              currentMode={currentMode}
              groupId={groupId ?? null}
              groupName={inspection.group?.name ?? null}
              standardName={inspection.standard?.name ?? null}
              sourceReady={hasSource}
              stationReady={stationReady}
              isLocked={inspection.isLocked}
              taskStage={inspection.taskStage ?? null}
              selectedClassesCount={selectedClassesCount}
            >
              <InspectionSourceControl
                currentMode={currentMode}
                cameraId={inspection.cameraId}
                file={inspection.file ?? null}
                disabled={inspection.isLocked}
                onCameraChange={inspection.setCameraId}
                onFileChange={inspection.setFile}
              />
            </StationTopbar>
          )}
        </SplitLayout.Topbar>

        <SplitLayout.Body bare>
          <div className={p.inspectBodyV32}>
            <Outlet context={inspection.outletContext} />
          </div>
        </SplitLayout.Body>
      </SplitLayout.Content>

      <SplitLayout.Panel>
        <div className={p.inspectSidePanelV32}>{renderPanel()}</div>
      </SplitLayout.Panel>
    </SplitLayout>
  );
}

function SelectionTopbar({
  currentMode,
  groupName,
  isLocked,
  taskStage,
  onModeChange,
}: {
  currentMode: InspectionModePath;
  groupName: string | null;
  isLocked: boolean;
  taskStage: string | null;
  onModeChange: (mode: InspectionModePath) => void;
}) {
  return (
    <div className={p.inspectSelectionTopbarV33}>
      <div className={p.inspectSelectionStatusV33}>
        <span><Activity /> Inspect station</span>
        <b><CircleDot /> Setup</b>
        {groupName ? <em>{groupName}</em> : <em>Выбор изделия</em>}
        {isLocked ? <strong><ListChecks /> {taskStage || "Running"}</strong> : null}
      </div>

      <div className={p.inspectModePillsV33}>
        {inspectionModePaths.map((mode) => (
          <button
            key={mode}
            type="button"
            data-active={mode === currentMode}
            onClick={() => onModeChange(mode)}
          >
            {modeLabel[mode]}
          </button>
        ))}
      </div>
    </div>
  );
}

function StationTopbar({
  currentMode,
  groupId,
  groupName,
  standardName,
  sourceReady,
  stationReady,
  isLocked,
  taskStage,
  selectedClassesCount,
  children,
}: {
  currentMode: InspectionModePath;
  groupId: string | null;
  groupName: string | null;
  standardName: string | null;
  sourceReady: boolean;
  stationReady: boolean;
  isLocked: boolean;
  taskStage: string | null;
  selectedClassesCount: number;
  children: React.ReactNode;
}) {
  return (
    <div className={p.inspectStationTopbarV34}>
      <div className={p.inspectStationContextV34}>
        <div className={p.inspectStationStatusV34}>
          <span><Activity /> Inspect station</span>
          <b data-ready={stationReady}>{stationReady ? <CheckCircle2 /> : <AlertTriangle />} {stationReady ? "Ready" : "Setup"}</b>
          {isLocked ? <em><ListChecks /> {taskStage || "Running"}</em> : null}
        </div>

        <div className={p.inspectStationCrumbsV34}>
          <strong>{modeLabel[currentMode]}</strong>
          <span>{groupName || "Project"}</span>
          <span>{standardName || "Reference"}</span>
        </div>

        <div className={p.inspectStationHintsV34}>
          <span data-ready={sourceReady}><FileImage /> {modeSourceLabel[currentMode]}</span>
          <span data-ready={selectedClassesCount > 0}><ShieldCheck /> {selectedClassesCount} classes</span>
          {groupId ? <Link to={paths.inspectionGroup(currentMode, groupId)}>Сменить reference</Link> : null}
          <Link to={paths.inspectionMode(currentMode)}>Сменить изделие</Link>
        </div>
      </div>

      <div className={p.inspectStationSourceV34}>{children}</div>
    </div>
  );
}
