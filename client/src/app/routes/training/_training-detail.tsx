import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import { useGetModels } from "@/page-components/models/api/get-models";
import { useGetTasks } from "@/page-components/tasks/api/get-tasks";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import { useTasksLive } from "@/page-components/tasks/hooks/use-task-live";
import type { GroupDetail, MlModel, TaskResponse } from "@/types/contracts";
import { Outlet, useLoaderData, useOutletContext } from "react-router-dom";

type TrainingModelOutletContext = {
  group: GroupDetail;
  models: MlModel[];
  tasks: TaskResponse[];
};

export const useTrainingModelOutletContext = () => useOutletContext<TrainingModelOutletContext>();

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const groupQuery = useGetGroup(groupId);
  const modelsQuery = useGetModels(groupId);
  const tasksQuery = useGetTasks(groupId);

  const group = groupQuery.data;
  const models = modelsQuery.data ?? [];
  const tasks = tasksQuery.data ?? [];

  const activeTrainingTaskIds = tasks.filter((task) => isActiveTaskStatus(task.status)).map((task) => task.id);
  useTasksLive({ taskIds: activeTrainingTaskIds, groupId });

  return (
    <QueryState
      size="page"
      isLoading={groupQuery.isLoading || modelsQuery.isLoading || tasksQuery.isLoading}
      isError={groupQuery.isError || modelsQuery.isError || tasksQuery.isError}
      isEmpty={!group}
      emptyTitle="Project not found"
      emptyDescription="Вернись к списку проектов и выбери изделие заново."
      errorTitle="Не удалось загрузить Train workspace"
    >
      {group ? <Outlet context={{ group, models, tasks }} /> : null}
    </QueryState>
  );
}
