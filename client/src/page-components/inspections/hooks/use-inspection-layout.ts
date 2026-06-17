import { getGroupQueryOptions } from "@/page-components/groups/api/get-group";
import {
  discardInspectionKeepalive,
  useDiscardInspection,
} from "@/page-components/inspections/api/discard-inspection";
import {
  stopRealtimeSessionKeepalive,
  useGetRealtimeStatus,
} from "@/page-components/inspections/api/realtime-session";
import {
  buildRunInspectionPayload,
  useRunInspection,
} from "@/page-components/inspections/api/run-inspection";
import type {
  InspectionOutletContext,
  InspectionRunOverrides,
} from "@/page-components/inspections/lib/inspection-context";
import { useGetTask } from "@/page-components/tasks/api/get-task";
import { getStandardQueryOptions } from "@/page-components/standards/api/get-standard";
import { useTaskLive } from "@/page-components/tasks/hooks/use-task-live";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import type { InspectionRealtimeStatus, InspectionTaskResult } from "@/types/contracts";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { useInspectionFocus } from "./use-inspection-focus";

type UseInspectionLayoutParams = {
  currentMode: string;
  groupId: string | null;
  standardId: string | null;
};

export function useInspectionLayout({
  currentMode,
  groupId,
  standardId,
}: UseInspectionLayoutParams) {
  const scopeKey = `${currentMode}:${groupId ?? ""}:${standardId ?? ""}`;

  const [file, setFile] = useState<File | null>(null);
  const [cameraId, setCameraId] = useState<string | null>(null);
  const [pendingTaskId, setPendingTaskId] = useState<string | null>(null);
  const [taskScopeKey, setTaskScopeKey] = useState<string | null>(null);
  const [realtimeSessionId, setRealtimeSessionId] = useState<string | null>(null);
  const [savedInspectionId, setSavedInspectionId] = useState<string | null>(null);
  const [selectedClassIds, setSelectedClassIds] = useState<string[]>([]);

  const groupQuery = useQuery({
    ...getGroupQueryOptions(groupId ?? ""),
    enabled: Boolean(groupId),
  });
  const group = groupQuery.data ?? null;

  const standardQuery = useQuery({
    ...getStandardQueryOptions(standardId ?? ""),
    enabled: Boolean(standardId),
  });
  const standard = standardQuery.data ?? null;

  const runMutation = useRunInspection({
    mutationConfig: {
      onSuccess: (data) => {
        setPendingTaskId(data.task_id);
        setTaskScopeKey(scopeKey);
        setSavedInspectionId(null);
      },
    },
  });

  const discardMutation = useDiscardInspection();

  const { data: task } = useGetTask(pendingTaskId);
  const { data: realtimeStatusData } = useGetRealtimeStatus(realtimeSessionId);
  useTaskLive({ taskId: pendingTaskId, groupId });
  const realtimeStatus = (realtimeStatusData ?? null) as InspectionRealtimeStatus | null;

  const isTaskInCurrentScope = taskScopeKey === scopeKey;
  const currentTask = isTaskInCurrentScope ? ((task ?? null) as typeof task | null) : null;
  const currentTaskStatus = currentTask?.status ?? null;
  const currentTaskStage = currentTask?.stage ?? null;
  const currentTaskProgress = currentTask?.progress_percent ?? null;

  const result = useMemo(
    () => (currentTask?.result as unknown as InspectionTaskResult | null | undefined) ?? null,
    [currentTask?.result]
  );

  const inspectedResult = currentMode === "realtime" && realtimeSessionId ? realtimeStatus : result;
  const focus = useInspectionFocus(inspectedResult);

  useEffect(() => {
    if (result?.inspection_id) {
      setSavedInspectionId(result.inspection_id);
    }
  }, [result?.inspection_id]);

  const taskNeedsDiscard =
    Boolean(currentTask) &&
    !savedInspectionId &&
    !result?.inspection_id &&
    (currentTaskStatus === "succeeded" || currentTaskStatus === "failed");

  const pendingTaskIdRef = useRef<string | null>(null);
  const realtimeSessionIdRef = useRef<string | null>(null);
  const taskNeedsDiscardRef = useRef(false);

  useEffect(() => {
    pendingTaskIdRef.current = pendingTaskId;
    realtimeSessionIdRef.current = realtimeSessionId;
    taskNeedsDiscardRef.current = taskNeedsDiscard;
  }, [pendingTaskId, realtimeSessionId, taskNeedsDiscard]);

  const resetInspectionState = () => {
    setPendingTaskId(null);
    setTaskScopeKey(null);
    setRealtimeSessionId(null);
    setSavedInspectionId(null);
    setFile(null);
  };

  useEffect(() => {
    const taskId = pendingTaskIdRef.current;
    const sessionId = realtimeSessionIdRef.current;

    if (taskId && taskNeedsDiscardRef.current) {
      discardMutation.mutate({ taskId });
    }
    if (sessionId) {
      stopRealtimeSessionKeepalive(sessionId);
    }

    resetInspectionState();
    setCameraId(null);
  }, [currentMode, groupId, standardId]);

  useEffect(() => {
    setSelectedClassIds([]);
  }, [groupId]);

  useEffect(() => {
    const handleBeforeUnload = () => {
      const taskId = pendingTaskIdRef.current;
      const sessionId = realtimeSessionIdRef.current;

      if (taskId && taskNeedsDiscardRef.current) {
        discardInspectionKeepalive(taskId);
      }
      if (sessionId) {
        stopRealtimeSessionKeepalive(sessionId);
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);

      const taskId = pendingTaskIdRef.current;
      const sessionId = realtimeSessionIdRef.current;

      if (taskId && taskNeedsDiscardRef.current) {
        discardInspectionKeepalive(taskId);
      }
      if (sessionId) {
        stopRealtimeSessionKeepalive(sessionId);
      }
    };
  }, []);

  const isAwaitingTask = isTaskInCurrentScope && Boolean(pendingTaskId) && !currentTask;
  const isLocked = runMutation.isPending || isAwaitingTask || isActiveTaskStatus(currentTaskStatus);

  const hasValidSource =
    currentMode === "photo"
      ? Boolean(file)
      : currentMode === "snapshot" || currentMode === "realtime"
        ? Boolean(cameraId)
        : false;

  const runDisabled =
    currentMode === "realtime" ||
    !groupId ||
    !standardId ||
    !hasValidSource ||
    selectedClassIds.length === 0 ||
    isLocked;

  const runLabel = isLocked
    ? currentTaskStage || "Проверка выполняется..."
    : currentMode === "snapshot"
      ? "Сделать снимок и проверить"
      : "Запустить проверку";

  const handleRun = (overrides: InspectionRunOverrides = {}) => {
    const nextFile = overrides.image !== undefined ? overrides.image : file;
    const nextCameraId = overrides.cameraId !== undefined ? overrides.cameraId : cameraId;

    const payloadResult = buildRunInspectionPayload({
      standard_id: standardId,
      selected_segment_class_ids: selectedClassIds,
      mode: currentMode,
      image: nextFile,
      camera_id: nextCameraId,
    });

    if (!payloadResult.ok) {
      return;
    }

    const previousTaskId = pendingTaskIdRef.current;
    const shouldDiscardPrevious = taskNeedsDiscardRef.current;

    runMutation.mutate(payloadResult.data, {
      onSuccess: () => {
        if (shouldDiscardPrevious && previousTaskId) {
          discardMutation.mutate({ taskId: previousTaskId });
        }
      },
    });
  };

  const outletContext: InspectionOutletContext = {
    currentMode,
    file,
    setFile,
    cameraId,
    setCameraId,
    selectedClassIds,
    result,
    realtimeSessionId,
    realtimeStatus,
    setRealtimeSessionId,
    taskStatus: currentTaskStatus,
    taskStage: currentTaskStage,
    taskProgress: currentTaskProgress,
    isLocked,
    runDisabled,
    runLabel,
    onRun: handleRun,
    focus,
  };

  return {
    file,
    setFile,
    taskStatus: currentTaskStatus,
    taskStage: currentTaskStage,
    taskProgress: currentTaskProgress,
    cameraId,
    focus,
    group,
    groupQuery,
    isLocked,
    outletContext,
    realtimeSessionId,
    realtimeStatus,
    resetInspectionState,
    result,
    savedInspectionId,
    selectedClassIds,
    setCameraId,
    setSavedInspectionId,
    setSelectedClassIds,
    standard,
    standardQuery,
  };
}
