import { ContentHeader } from "@/components/layouts/content-header/content-header";
import { Section } from "@/components/layouts/section/section";
import { Badge } from "@/components/ui/badge/badge";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import { CoverageItem } from "@/page-components/groups/components/coverage-item/coverage-item";
import { useGetModels } from "@/page-components/models/api/get-models";
import { ExportModel } from "@/page-components/models/components/export-model/export-model";
import { ImportModel } from "@/page-components/models/components/import-model/import-model";
import { TrainModel } from "@/page-components/models/components/train-model/train-model";
import { useGetTasks } from "@/page-components/tasks/api/get-tasks";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import { useTasksLive } from "@/page-components/tasks/hooks/use-task-live";
import { GroupDetail, MlModel, TaskResponse } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { Outlet, useLoaderData, useOutletContext } from "react-router-dom";

type TrainingModelOutletContext = {
  group: GroupDetail;
  models: MlModel[];
  tasks: TaskResponse[];
};

export const useTrainingModelOutletContext = () => {
  return useOutletContext<TrainingModelOutletContext>();
};

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: group } = useGetGroup(groupId);
  const { data: models } = useGetModels(groupId);
  const { data: tasks } = useGetTasks(groupId);

  const activeTrainingTaskIds = tasks
    .filter((task) => isActiveTaskStatus(task.status))
    .map((task) => task.id);

  useTasksLive({ taskIds: activeTrainingTaskIds, groupId });

  const hasMinimumTrainingData =
    group.stats.standards_count > 0 &&
    group.stats.images_count > 0 &&
    group.stats.annotated_images_count > 0 &&
    group.stats.segment_classes_count > 0;

  const hasActiveTrainingTask = tasks.some((task) => isActiveTaskStatus(task.status));

  return (
    <>
      <ContentHeader>
        <ContentHeader.Top
          title={group.name}
          subtitles={[
            ...(group.description ? [group.description] : []),
            `Создана ${formatDate(group.created_at)}`,
          ]}
          meta={[
            `${group.stats.standards_count} эталонов`,
            `${group.stats.images_count} изображений`,
            `${group.stats.annotated_images_count} размечено`,
            `${group.stats.polygons_count} аннотаций`,
            `${group.stats.segment_classes_count} классов`,
          ]}
        >
          <ContentHeader.Actions>
            <ImportModel groupId={group.id} />
            <ExportModel models={models} />
            <TrainModel
              groupId={group.id}
              canTrain={hasMinimumTrainingData}
              isTrainingLocked={hasActiveTrainingTask}
            />
          </ContentHeader.Actions>
        </ContentHeader.Top>
      </ContentHeader>

      <Section
        title="Покрытие эталонов"
        side={
          <Badge>
            {group.stats.annotated_images_count} / {group.stats.images_count}
          </Badge>
        }
        bordered
        scrollable
        maxContentHeight={280}
      >
        <QueryState
          size="block"
          isEmpty={group.standards.length === 0}
          emptyTitle="Нет эталонов"
          emptyDescription="Создайте эталоны для этой группы"
        >
          {group.standards.map((std) => (
            <CoverageItem key={std.id} standard={std} />
          ))}
        </QueryState>
      </Section>

      <Outlet context={{ group, models, tasks }} />
    </>
  );
}
