import { paths, type InspectionModePath } from "@/app/paths";
import { SplitLayout } from "@/components/layouts/split-layout/split-layout";
import QueryState from "@/components/ui/query-state/query-state";
import { ClassSelector } from "@/page-components/inspections/components/class-selector/class-selector";
import { InspectionResultPanel } from "@/page-components/inspections/components/inspection-result-panel/inspection-result-panel";
import { InspectionTopbar } from "@/page-components/inspections/components/inspection-topbar/inspection-topbar";
import { useInspectionLayout } from "@/page-components/inspections/hooks/use-inspection-layout";
import { Outlet, useLoaderData, useNavigate, useParams } from "react-router-dom";

export function Component() {
  const { mode: currentMode } = useLoaderData() as { mode: InspectionModePath };
  const { groupId, standardId } = useParams();
  const navigate = useNavigate();
  const inspection = useInspectionLayout({
    currentMode,
    groupId: groupId ?? null,
    standardId: standardId ?? null,
  });

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
        <QueryState
          isEmpty
          size="block"
          emptyTitle="Выберите группу"
          emptyDescription="После выбора группы здесь появятся классы проверки"
        />
      );
    }

    if (inspection.result) {
      return (
        <InspectionResultPanel
          result={inspection.result}
          savedInspectionId={inspection.savedInspectionId}
          onSaved={inspection.setSavedInspectionId}
          onDiscarded={() => {
            inspection.resetInspectionState();
          }}
          activeMatchKey={inspection.focus.activeMatchKey}
          setActiveMatchKey={inspection.focus.setActiveMatchKey}
        />
      );
    }

    if (inspection.groupQuery.isPending) {
      return (
        <QueryState
          isLoading
          size="block"
          loadingText="Подготавливаем классы для проверки"
        />
      );
    }

    if (inspection.groupQuery.isError) {
      return (
        <QueryState
          isError
          size="block"
          errorTitle="Не удалось загрузить группу"
          errorDescription="Проверьте выбранную группу и попробуйте снова"
        />
      );
    }

    if (!inspection.group) {
      return (
        <QueryState
          isEmpty
          size="block"
          emptyTitle="Группа не найдена"
          emptyDescription="Проверьте выбранную группу и попробуйте снова"
        />
      );
    }

    return (
      <ClassSelector
        group={inspection.group}
        value={inspection.selectedClassIds}
        onChange={inspection.setSelectedClassIds}
        disabled={inspection.isLocked}
      />
    );
  };

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

  const handleGroupChange = (nextGroupId: string | null) => {
    if (!nextGroupId) {
      navigate(paths.inspectionMode(currentMode));
      return;
    }

    navigate(paths.inspectionGroup(currentMode, nextGroupId));
  };

  const handleStandardChange = (nextStandardId: string | null) => {
    if (!groupId) {
      return;
    }

    if (!nextStandardId) {
      navigate(paths.inspectionGroup(currentMode, groupId));
      return;
    }

    navigate(paths.inspectionStandard(currentMode, groupId, nextStandardId));
  };

  return (
    <SplitLayout>
      <SplitLayout.Content>
        <SplitLayout.Topbar>
          <InspectionTopbar
            currentMode={currentMode}
            groupId={groupId ?? null}
            standardId={standardId ?? null}
            cameraId={inspection.cameraId}
            onModeChange={handleModeChange}
            onGroupChange={handleGroupChange}
            onStandardChange={handleStandardChange}
            onCameraChange={inspection.setCameraId}
          />
        </SplitLayout.Topbar>

        <SplitLayout.Body bare>
          <Outlet context={inspection.outletContext} />
        </SplitLayout.Body>
      </SplitLayout.Content>

      <SplitLayout.Panel>{renderPanel()}</SplitLayout.Panel>
    </SplitLayout>
  );
}
