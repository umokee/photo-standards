import { InspectionWorkspace } from "@/page-components/inspections/components/inspection-workspace/inspection-workspace";
import type { InspectionOutletContext } from "@/page-components/inspections/lib/inspection-context";
import { useLoaderData, useOutletContext } from "react-router-dom";

export function Component() {
  const { standardId } = useLoaderData() as { standardId: string };
  const ctx = useOutletContext<InspectionOutletContext>();

  return (
    <InspectionWorkspace
      standardId={standardId}
      currentMode={ctx.currentMode}
      file={ctx.file}
      onFileChange={ctx.setFile}
      cameraId={ctx.cameraId}
      result={ctx.result}
      realtimeSessionId={ctx.realtimeSessionId}
      realtimeStatus={ctx.realtimeStatus}
      setRealtimeSessionId={ctx.setRealtimeSessionId}
      taskStatus={ctx.taskStatus}
      taskStage={ctx.taskStage}
      taskProgress={ctx.taskProgress}
      isLocked={ctx.isLocked}
      selectedClassIds={ctx.selectedClassIds}
      activeMatchKey={ctx.focus.activeMatchKey}
      onActiveMatchChange={ctx.focus.setActiveMatchKey}
      runDisabled={ctx.runDisabled}
      runLabel={ctx.runLabel}
      onRun={ctx.onRun}
    />
  );
}
