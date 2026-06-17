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
import { type ReactNode } from "react";
import { Link, Outlet, useLoaderData, useNavigate, useParams } from "react-router-dom";
import s from "./_inspect-strict.module.scss";

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
        <div className={s.blockedPanel}>
          <MousePointer2 />
          <h3>Выбери изделие</h3>
          <p>На первом шаге справа нет классов или эталонов. Выбор проекта находится в основной области.</p>
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
        <div className={s.blockedPanel}>
          <Layers3 />
          <h3>Выбери reference</h3>
          <p>Классы и запуск проверки появятся после выбора эталонного вида.</p>
          <div className={s.blockedStats}>
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
      <div className={s.panelInner}>
        <div className={s.panelHead}>
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
          <div className={s.stationBody}>
            <Outlet context={inspection.outletContext} />
          </div>
        </SplitLayout.Body>
      </SplitLayout.Content>

      <SplitLayout.Panel>
        <div className={s.sidePanel}>{renderPanel()}</div>
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
    <div className={s.selectionTopbar}>
      <div className={s.selectionStatus}>
        <span><Activity /> Inspect station</span>
        <b><CircleDot /> Setup</b>
        <em>{groupName || "Выбор изделия"}</em>
        {isLocked ? <strong><ListChecks /> {taskStage || "Running"}</strong> : null}
      </div>

      <div className={s.modePills}>
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
  children: ReactNode;
}) {
  return (
    <div className={s.stationTopbar}>
      <div className={s.stationContext}>
        <div className={s.stationStatus}>
          <span><Activity /> Inspect station</span>
          <b data-state={stationReady ? "ready" : "setup"}>
            {stationReady ? <CheckCircle2 /> : <AlertTriangle />} {stationReady ? "Ready" : "Setup"}
          </b>
          {isLocked ? <em><ListChecks /> {taskStage || "Running"}</em> : null}
        </div>

        <div className={s.stationCrumbs}>
          <strong>{modeLabel[currentMode]}</strong>
          <span>{groupName || "Project"}</span>
          <span>{standardName || "Reference"}</span>
        </div>

        <div className={s.stationHints}>
          <span data-state={sourceReady ? "ready" : "setup"}><FileImage /> {modeSourceLabel[currentMode]}</span>
          <span data-state={selectedClassesCount > 0 ? "ready" : "setup"}><ShieldCheck /> {selectedClassesCount} classes</span>
          {groupId ? <Link to={paths.inspectionGroup(currentMode, groupId)}>Сменить reference</Link> : null}
          <Link to={paths.inspectionMode(currentMode)}>Сменить изделие</Link>
        </div>
      </div>

      <div className={s.sourceSlot}>{children}</div>
    </div>
  );
}
