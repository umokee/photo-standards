import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import { useGetModels } from "@/page-components/models/api/get-models";
import { ExportModel } from "@/page-components/models/components/export-model/export-model";
import { ImportModel } from "@/page-components/models/components/import-model/import-model";
import { TrainModel } from "@/page-components/models/components/train-model/train-model";
import { useGetTasks } from "@/page-components/tasks/api/get-tasks";
import { useTasksLive } from "@/page-components/tasks/hooks/use-task-live";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import { GroupDetail, MlModel, TaskResponse } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { Box, Brain, Image, Upload } from "lucide-react";
import { Outlet, useLoaderData, useOutletContext } from "react-router-dom";
import p from "../platform-pages.module.scss";

type TrainingModelOutletContext = { group: GroupDetail; models: MlModel[]; tasks: TaskResponse[] };
export const useTrainingModelOutletContext = () => useOutletContext<TrainingModelOutletContext>();

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: group } = useGetGroup(groupId);
  const { data: models } = useGetModels(groupId);
  const { data: tasks } = useGetTasks(groupId);

  const activeTrainingTaskIds = tasks.filter((task) => isActiveTaskStatus(task.status)).map((task) => task.id);
  useTasksLive({ taskIds: activeTrainingTaskIds, groupId });

  const hasMinimumTrainingData = group.stats.standards_count > 0 && group.stats.images_count > 0 && group.stats.annotated_images_count > 0 && group.stats.segment_classes_count > 0;
  const hasActiveTrainingTask = tasks.some((task) => isActiveTaskStatus(task.status));

  return (
    <div className={p.page}>
      <section className={p.datasetHeader}>
        <div className={p.datasetThumb} />
        <div className={p.datasetTitle}>
          <h1>{group.name}</h1>
          <div className={p.metaLine}>
            <span><Brain /> {models.length} models</span>
            <span><Image /> {group.stats.images_count} images</span>
            <span><Box /> {group.stats.segment_classes_count} classes</span>
            <span>Updated {formatDate(group.created_at)}</span>
          </div>
          <p>Train new YOLO model from annotated standards or import existing .pt weights.</p>
        </div>
        <div className={p.headerActions}>
          <ImportModel groupId={group.id} />
          <ExportModel models={models} />
          <TrainModel groupId={group.id} canTrain={hasMinimumTrainingData} isTrainingLocked={hasActiveTrainingTask} />
        </div>
      </section>

      <div className={p.grid2}>
        <section className={p.panelCard}>
          <div className={p.cardTitleRow}><div><h3>Models</h3><p>Training runs and imported weights.</p></div></div>
          <QueryState isEmpty={!models.length} size="block" emptyTitle="No models" emptyDescription="Train or import the first model for this dataset.">
            <div className={p.tableLike}>
              <div className={p.tableHead}><span>Name</span><span>Image size</span><span>Classes</span><span>Status</span><span>Created</span></div>
              {models.map((model) => (
                <div className={p.tableRow} key={model.id}>
                  <span>{model.architecture} {model.version ? `v${model.version}` : ""}</span>
                  <span>{model.imgsz}</span>
                  <span>{model.num_classes ?? "—"}</span>
                  <span><b className={model.is_active ? p.ok : undefined}>{model.is_active ? "Active" : "Ready"}</b></span>
                  <span>{formatDate(model.created_at)}</span>
                </div>
              ))}
            </div>
          </QueryState>
        </section>
        <aside className={p.sidePanel}>
          <h3>Dataset readiness</h3>
          <p>Для обучения нужны изображения, классы и размеченные полигоны.</p>
          <div className={p.dropzonePreview}><Upload /><span>{hasMinimumTrainingData ? "Ready for training" : "Not enough annotation data"}</span><small>{group.stats.annotated_images_count}/{group.stats.images_count} labeled images</small></div>
        </aside>
      </div>

      <Outlet context={{ group, models, tasks }} />
    </div>
  );
}
