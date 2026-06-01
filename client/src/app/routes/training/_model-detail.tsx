import { paths } from "@/app/paths";
import { Section } from "@/components/layouts/section/section";
import QueryState from "@/components/ui/query-state/query-state";
import { useActivateModel } from "@/page-components/models/api/activate-model";
import { useDeleteModel } from "@/page-components/models/api/delete-model";
import { ModelCard } from "@/page-components/models/components/model-card/model-card";
import { getModelTask } from "@/page-components/models/lib/model-helpers";
import { useCancelTask } from "@/page-components/tasks/api/cancel-task";
import { usePauseTask } from "@/page-components/tasks/api/pause-task";
import { useResumeTask } from "@/page-components/tasks/api/resume-task";
import { useLoaderData, useNavigate } from "react-router-dom";
import { useTrainingModelOutletContext } from "./_training-detail";

export function Component() {
  const navigate = useNavigate();
  const { modelId } = useLoaderData() as { modelId: string };
  const { group, models, tasks } = useTrainingModelOutletContext();

  const activateMutation = useActivateModel({ groupId: group.id });
  const deleteMutation = useDeleteModel({ groupId: group.id });
  const cancelMutation = useCancelTask({ groupId: group.id });
  const pauseMutation = usePauseTask({ groupId: group.id });
  const resumeMutation = useResumeTask({ groupId: group.id });

  const toggleModel = (targetModelId: string) => {
    if (modelId === targetModelId) {
      navigate(paths.trainingGroup(group.id));
      return;
    }

    navigate(paths.trainingModel(group.id, targetModelId));
  };

  return (
    <QueryState
      size="block"
      isEmpty={group.stats.models_count === 0}
      emptyTitle="Нет моделей"
      emptyDescription="Обучите модель для этой группы"
    >
      <Section title={`Модели · ${group.stats.models_count}`} scrollable>
        {models.map((model) => {
          const isExpanded = modelId === model.id;
          const task = getModelTask(model, tasks);

          return (
            <ModelCard
              key={model.id}
              model={model}
              task={task}
              expanded={isExpanded}
              onToggle={() => toggleModel(model.id)}
              onActivate={(targetModelId) => activateMutation.mutate(targetModelId)}
              onDelete={(targetModelId) => {
                deleteMutation.mutate(targetModelId, {
                  onSuccess: () => {
                    if (targetModelId === modelId) {
                      navigate(paths.trainingGroup(group.id));
                    }
                  },
                });
              }}
              onPause={(taskId) => pauseMutation.mutate(taskId)}
              onResume={(taskId) => resumeMutation.mutate(taskId)}
              onCancel={(taskId) => cancelMutation.mutate(taskId)}
              isActivating={activateMutation.isPending && activateMutation.variables === model.id}
              isDeleting={deleteMutation.isPending && deleteMutation.variables === model.id}
              isPausing={pauseMutation.isPending && pauseMutation.variables === task?.id}
              isResuming={resumeMutation.isPending && resumeMutation.variables === task?.id}
              isCancelling={cancelMutation.isPending && cancelMutation.variables === task?.id}
            />
          );
        })}
      </Section>
    </QueryState>
  );
}
