import { queryKeys } from "@/lib/query-keys";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { isTerminalTaskStatus } from "../lib/task-helpers";

type TaskEvent =
  | {
      kind: "task";
      task_id: string;
      event: "status";
      status: string;
      stage: string | null;
      message: string | null;
      error: string | null;
    }
  | {
      kind: "task";
      task_id: string;
      event: "progress";
      current: number;
      total: number;
      percent: number | null;
      stage: string;
    }
  | { kind: "task"; task_id: string; event: "heartbeat" };

type UseTaskLiveParams = {
  taskId: string | null | undefined;
  groupId?: string | null | undefined;
  modelId?: string | null | undefined;
};

type UseTasksLiveParams = {
  taskIds: Array<string | null | undefined>;
  groupId?: string | null | undefined;
};

const isTerminalStatusEvent = (
  event: TaskEvent
): event is Extract<TaskEvent, { event: "status" }> => {
  return event.event === "status" && isTerminalTaskStatus(event.status);
};

export function useTaskLive({ taskId, groupId, modelId }: UseTaskLiveParams) {
  const qc = useQueryClient();

  useEffect(() => {
    if (!taskId) return;

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/tasks/${taskId}`);

    ws.onmessage = (msg) => {
      const event = JSON.parse(msg.data) as TaskEvent;
      if (event.event === "heartbeat") return;

      qc.invalidateQueries({ queryKey: queryKeys.training.task(taskId) });
      qc.invalidateQueries({ queryKey: ["training", "metrics-history"] });

      if (groupId && event.event === "status") {
        qc.invalidateQueries({ queryKey: queryKeys.training.tasks(groupId) });
      }

      if (isTerminalStatusEvent(event)) {
        if (groupId) {
          qc.invalidateQueries({ queryKey: queryKeys.training.models(groupId) });
        }
        if (modelId) {
          qc.invalidateQueries({ queryKey: queryKeys.training.model(modelId) });
        }
      }
    };

    return () => ws.close();
  }, [taskId, groupId, modelId, qc]);
}

export function useTasksLive({ taskIds, groupId }: UseTasksLiveParams) {
  const qc = useQueryClient();
  const ids = [...new Set(taskIds.filter((taskId): taskId is string => !!taskId))].sort();
  const idsKey = ids.join("|");

  useEffect(() => {
    if (ids.length === 0) return;

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const sockets = ids.map((taskId) => {
      const ws = new WebSocket(`${proto}://${window.location.host}/ws/tasks/${taskId}`);

      ws.onmessage = (msg) => {
        const event = JSON.parse(msg.data) as TaskEvent;
        if (event.event === "heartbeat") return;

        qc.invalidateQueries({ queryKey: queryKeys.training.task(taskId) });
        qc.invalidateQueries({ queryKey: ["training", "metrics-history"] });

        if (groupId && event.event === "status") {
          qc.invalidateQueries({ queryKey: queryKeys.training.tasks(groupId) });
        }

        if (groupId && isTerminalStatusEvent(event)) {
          qc.invalidateQueries({ queryKey: queryKeys.training.models(groupId) });
        }
      };

      return ws;
    });

    return () => {
      for (const ws of sockets) {
        ws.close();
      }
    };
  }, [groupId, idsKey, qc]);
}
