import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useActivateModel } from "@/page-components/models/api/activate-model";
import { useDeleteModel } from "@/page-components/models/api/delete-model";
import { ExportModel } from "@/page-components/models/components/export-model/export-model";
import { ImportModel } from "@/page-components/models/components/import-model/import-model";
import { TrainModel } from "@/page-components/models/components/train-model/train-model";
import { getModelTask } from "@/page-components/models/lib/model-helpers";
import { useCancelTask } from "@/page-components/tasks/api/cancel-task";
import { usePauseTask } from "@/page-components/tasks/api/pause-task";
import { useResumeTask } from "@/page-components/tasks/api/resume-task";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import type { MlModel } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Boxes,
  Brain,
  CalendarClock,
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  Database,
  FileJson,
  Filter,
  GitBranch,
  Image,
  Layers3,
  Pause,
  Play,
  RefreshCw,
  Ruler,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Tags,
  Trash2,
  UploadCloud,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { Link, useLoaderData, useNavigate } from "react-router-dom";
import p from "../platform-pages.module.scss";
import { useTrainingModelOutletContext } from "./_training-detail";

const metricNames = ["mAP50_95", "mAP50", "precision", "recall"] as const;
type MetricName = typeof metricNames[number];
type ModelFilter = "all" | "active" | "trained" | "draft" | "withTask";
type SortMode = "created" | "score" | "name";
type SeriesSource = "history" | "summary" | "todo";
type FlexibleModel = MlModel & Record<string, unknown>;
type TaskLike = NonNullable<ReturnType<typeof getModelTask>>;
type FlexibleTask = TaskLike & Record<string, unknown>;

type ChartDefinition = {
  key: string;
  title: string;
  kind: "metric" | "loss" | "lr";
  aliases: string[];
};

type SeriesForModel = {
  model: MlModel;
  values: number[];
  source: SeriesSource;
};

const metricCharts: ChartDefinition[] = [
  { key: "precision", title: "precision", kind: "metric", aliases: ["precision", "precision(B)", "metrics/precision(B)", "metrics.precision"] },
  { key: "recall", title: "recall", kind: "metric", aliases: ["recall", "recall(B)", "metrics/recall(B)", "metrics.recall"] },
  { key: "mAP50", title: "mAP50", kind: "metric", aliases: ["mAP50", "mAP50(B)", "metrics/mAP50(B)", "metrics.mAP50"] },
  { key: "mAP50_95", title: "mAP50-95", kind: "metric", aliases: ["mAP50_95", "mAP50-95", "mAP50-95(B)", "metrics/mAP50-95(B)", "metrics.mAP50_95"] },
];

const lossCharts: ChartDefinition[] = [
  { key: "train_box_loss", title: "box loss", kind: "loss", aliases: ["train/box_loss", "box_loss", "train_box_loss"] },
  { key: "train_cls_loss", title: "class loss", kind: "loss", aliases: ["train/cls_loss", "cls_loss", "train_cls_loss"] },
  { key: "train_dfl_loss", title: "dfl loss", kind: "loss", aliases: ["train/dfl_loss", "dfl_loss", "train_dfl_loss"] },
  { key: "val_box_loss", title: "val box loss", kind: "loss", aliases: ["val/box_loss", "val_box_loss"] },
  { key: "val_cls_loss", title: "val class loss", kind: "loss", aliases: ["val/cls_loss", "val_cls_loss"] },
];

const lrCharts: ChartDefinition[] = [
  { key: "lr0", title: "learning rate 0", kind: "lr", aliases: ["lr/pg0", "lr0", "lr_pg0"] },
  { key: "lr1", title: "learning rate 1", kind: "lr", aliases: ["lr/pg1", "lr1", "lr_pg1"] },
  { key: "lr2", title: "learning rate 2", kind: "lr", aliases: ["lr/pg2", "lr2", "lr_pg2"] },
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asNumber(value: unknown): number | null {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  return value;
}

function normalizeSeries(value: unknown, alias: string): number[] | null {
  if (Array.isArray(value)) {
    const values = value
      .map((item) => {
        if (typeof item === "number") return item;
        if (!isRecord(item)) return null;
        return asNumber(item.value) ?? asNumber(item.y) ?? asNumber(item[alias]) ?? asNumber(item.metric);
      })
      .filter((item): item is number => item !== null);
    return values.length > 1 ? values : null;
  }

  if (isRecord(value)) {
    if (Array.isArray(value.values)) return normalizeSeries(value.values, alias);
    if (Array.isArray(value.data)) return normalizeSeries(value.data, alias);
    if (Array.isArray(value.points)) return normalizeSeries(value.points, alias);
  }

  return null;
}

function readSeriesBucket(bucket: unknown, aliases: string[]): number[] | null {
  if (!isRecord(bucket)) return null;

  for (const alias of aliases) {
    const direct = normalizeSeries(bucket[alias], alias);
    if (direct) return direct;
  }

  for (const alias of aliases) {
    const dotted = alias.split(".").reduce<unknown>((cursor, part) => (isRecord(cursor) ? cursor[part] : undefined), bucket);
    const nested = normalizeSeries(dotted, alias);
    if (nested) return nested;
  }

  return null;
}

function scalarMetric(model: MlModel, key: string): number | null {
  const metrics = model.metrics as Record<string, unknown> | undefined;
  const value = metrics?.[key] ?? metrics?.[key.replace("_", "-")];
  const number = asNumber(value);
  if (number == null) return null;
  return number <= 1 ? Math.max(0, Math.min(1, number)) : Math.max(0, Math.min(1, number / 100));
}

function estimatedMetricCurve(finalValue: number) {
  const target = Math.max(0.02, Math.min(0.99, finalValue));
  return Array.from({ length: 30 }, (_, index) => {
    const t = index / 29;
    const noise = Math.sin(index * 1.7) * 0.014;
    return Math.max(0, Math.min(1, target * (1 - Math.exp(-4.2 * t)) + noise));
  });
}

function seriesForChart(model: MlModel, task: TaskLike | null, chart: ChartDefinition): SeriesForModel {
  const flexibleModel = model as FlexibleModel;
  const flexibleTask = task as FlexibleTask | null;
  const buckets = [
    flexibleModel.metrics_history,
    flexibleModel.history,
    flexibleModel.curves,
    flexibleModel.training_history,
    flexibleModel.training_metrics,
    flexibleModel.results,
    flexibleModel.train_results,
    flexibleTask?.metrics_history,
    flexibleTask?.history,
    flexibleTask?.curves,
    flexibleTask?.metrics,
  ];

  for (const bucket of buckets) {
    const values = readSeriesBucket(bucket, chart.aliases);
    if (values) return { model, values, source: "history" };
  }

  if (chart.kind === "metric") {
    const scalar = scalarMetric(model, chart.key);
    if (scalar != null) return { model, values: estimatedMetricCurve(scalar), source: "summary" };
  }

  return { model, values: [], source: "todo" };
}

function formatMetric(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "—";
  return value <= 1 ? `${Math.round(value * 100)}%` : String(Math.round(value * 100) / 100);
}

function formatChartValue(value: number | null | undefined, kind: ChartDefinition["kind"]) {
  if (value == null || Number.isNaN(value)) return "—";
  if (kind === "metric") return formatMetric(value);
  if (kind === "lr") return value.toPrecision(3);
  return value.toFixed(value < 1 ? 3 : 2);
}

function metricNumber(model: MlModel, name: MetricName) {
  return scalarMetric(model, name) ?? 0;
}

function modelScore(model: MlModel) {
  return metricNumber(model, "mAP50_95") || metricNumber(model, "mAP50") || metricNumber(model, "precision") || metricNumber(model, "recall");
}

function modelTitle(model: MlModel) {
  return `${model.architecture} ${model.version ? `v${model.version}` : ""}`.trim();
}

function modelFamily(model: MlModel) {
  const meta = model as MlModel & { task_type?: string; model_type?: string; type?: string };
  const raw = (meta.task_type ?? meta.model_type ?? meta.type ?? "").toLowerCase();
  if (raw.includes("seg")) return "Segment";
  if (raw.includes("detect")) return "Detect";
  if (model.architecture.toLowerCase().includes("seg")) return "Segment";
  return "Detect";
}

function splitTotal(model: MlModel) {
  return (model.train_count ?? 0) + (model.val_count ?? 0) + (model.test_count ?? 0);
}

function totalImages(model: MlModel) {
  return (model.total_images ?? splitTotal(model)) || 0;
}

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

export function Component() {
  const navigate = useNavigate();
  const { modelId } = useLoaderData() as { modelId: string | null };
  const { group, models, tasks } = useTrainingModelOutletContext();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<ModelFilter>("all");
  const [sort, setSort] = useState<SortMode>("created");
  const [railOpen, setRailOpen] = useState(true);
  const [compareIds, setCompareIds] = useState<string[] | null>(null);

  const activateMutation = useActivateModel({ groupId: group.id });
  const deleteMutation = useDeleteModel({ groupId: group.id });
  const cancelMutation = useCancelTask({ groupId: group.id });
  const pauseMutation = usePauseTask({ groupId: group.id });
  const resumeMutation = useResumeTask({ groupId: group.id });

  const filteredModels = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const result = models.filter((model) => {
      const task = getModelTask(model, tasks);
      const matchesQuery = !needle || [modelTitle(model), model.id, model.architecture, model.weights_path ?? ""].some((value) => value.toLowerCase().includes(needle));
      const matchesFilter =
        filter === "all" ? true :
        filter === "active" ? model.is_active :
        filter === "trained" ? Boolean(model.trained_at) :
        filter === "draft" ? !model.trained_at :
        Boolean(task);
      return matchesQuery && matchesFilter;
    });

    return [...result].sort((a, b) => {
      if (sort === "score") return modelScore(b) - modelScore(a);
      if (sort === "name") return modelTitle(a).localeCompare(modelTitle(b));
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });
  }, [filter, models, query, sort, tasks]);

  const selectedModel =
    models.find((model) => model.id === modelId) ??
    models.find((model) => model.is_active) ??
    filteredModels[0] ??
    models[0] ??
    null;

  useEffect(() => {
    if (compareIds !== null || !models.length) return;
    const active = models.find((model) => model.is_active);
    const initial = [active, ...models.filter((model) => model.id !== active?.id)]
      .filter(Boolean)
      .slice(0, 3)
      .map((model) => model.id);
    setCompareIds(initial);
  }, [compareIds, models]);

  const selectedCompareIds = compareIds ?? [];
  const compareModels = selectedCompareIds
    .map((id) => models.find((model) => model.id === id))
    .filter((model): model is MlModel => Boolean(model));

  const chartModels = compareModels.length ? compareModels : filteredModels.slice(0, 3);

  const groupedModels = useMemo(() => {
    const result = new Map<string, MlModel[]>();
    filteredModels.forEach((model) => {
      const family = modelFamily(model);
      result.set(family, [...(result.get(family) ?? []), model]);
    });
    return Array.from(result.entries());
  }, [filteredModels]);

  const canTrain =
    group.stats.standards_count > 0 &&
    group.stats.images_count > 0 &&
    group.stats.annotated_images_count > 0 &&
    group.stats.segment_classes_count > 0;
  const hasActiveTrainingTask = tasks.some((task) => isActiveTaskStatus(task.status));

  const toggleCompare = (model: MlModel) => {
    setCompareIds((current) => {
      const ids = current ?? [];
      if (ids.includes(model.id)) return ids.filter((id) => id !== model.id);
      return [...ids, model.id].slice(-4);
    });
  };

  return (
    <div className={`${p.page} ${p.trainPageV27} ${p.trainModelsScreenV31}`}>
      <header className={p.trainHeaderV31}>
        <div>
          <span className={p.eyebrow}><Brain /> Train / Models</span>
          <h1>Model lab</h1>
          <p>Графики, метрики, выбор active model и metadata датасета. Списки и графики скроллятся внутри workspace.</p>
        </div>
        <div className={p.trainActionsV27}>
          <ImportModel groupId={group.id} />
          <ExportModel models={models} />
          <TrainModel groupId={group.id} canTrain={canTrain} isTrainingLocked={hasActiveTrainingTask} />
        </div>
      </header>

      <section className={p.modelLabShellV31}>
        <aside className={`${p.modelCompareRailV31} ${railOpen ? "" : p.modelCompareRailCollapsedV31}`}>
          <div className={p.modelRailTopV31}>
            <label className={p.modelSearchV31}>
              <Search />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models..." />
            </label>

            <button className={p.modelRailCollapseV31} type="button" onClick={() => setRailOpen((value) => !value)}>
              {railOpen ? "Collapse" : "Models"}
            </button>
          </div>

          {railOpen ? (
            <>
              <div className={p.modelRailCardV31}>
                <div className={p.modelRailTitleV31}>
                  <span><Boxes /> Models</span>
                  <b>{selectedCompareIds.length} selected</b>
                </div>
                <button type="button" className={p.modelMiniActionV31} onClick={() => setCompareIds([])}>Deselect all</button>

                <div className={p.modelFilterRowV31}>
                  <Filter />
                  {(["all", "active", "trained", "draft", "withTask"] as ModelFilter[]).map((value) => (
                    <button key={value} type="button" className={filter === value ? p.modelFilterActiveV31 : ""} onClick={() => setFilter(value)}>
                      {value === "withTask" ? "jobs" : value}
                    </button>
                  ))}
                </div>
              </div>

              <div className={p.modelRailBodyV31}>
                <QueryState
                  size="block"
                  isEmpty={!filteredModels.length}
                  emptyTitle="No models"
                  emptyDescription="Train/import a model or clear filters."
                >
                  <div className={p.modelRailListV31}>
                    {groupedModels.map(([family, items]) => (
                      <section key={family} className={p.modelFamilyV31}>
                        <header><ChevronDown /><span>{family}</span><b>{items.length}</b></header>
                        {items.map((model) => {
                          const task = getModelTask(model, tasks);
                          const isCurrent = selectedModel?.id === model.id;
                          const checked = selectedCompareIds.includes(model.id);
                          const status = model.is_active ? "active" : task?.status ?? (model.trained_at ? "trained" : "draft");

                          return (
                            <div
                              key={model.id}
                              role="button"
                              tabIndex={0}
                              className={`${p.modelOptionV31} ${isCurrent ? p.modelOptionCurrentV31 : ""} ${checked ? p.modelOptionCheckedV31 : ""}`}
                              onClick={() => navigate(paths.trainingModel(group.id, model.id))}
                              onKeyDown={(event) => {
                                if (event.key === "Enter") navigate(paths.trainingModel(group.id, model.id));
                              }}
                            >
                              <button
                                type="button"
                                className={p.modelCheckV31}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  toggleCompare(model);
                                }}
                                aria-label="Toggle model comparison"
                              >
                                {checked ? <CheckCircle2 /> : <CircleDashed />}
                              </button>
                              <i />
                              <div>
                                <strong>{modelTitle(model)}</strong>
                                <span>{Math.round(modelScore(model) * 100)}% · {model.weights_size_mb ?? "—"} MB</span>
                              </div>
                              <em>{status}</em>
                            </div>
                          );
                        })}
                      </section>
                    ))}
                  </div>
                </QueryState>
              </div>

              <div className={p.modelDropZoneV31}>
                <UploadCloud />
                <strong>Drop .pt model files</strong>
                <span>TODO backend: import should parse checkpoint/results artifacts and fill metrics history.</span>
              </div>
            </>
          ) : null}
        </aside>

        <main className={p.modelAnalysisPaneV31}>
          <div className={p.modelAnalysisToolbarV31}>
            <div>
              <span className={p.eyebrow}><BarChart3 /> Metrics</span>
              <h2>Training graphs</h2>
            </div>

            <div className={p.modelSortV31}>
              <SlidersHorizontal />
              <span>Sort</span>
              <select value={sort} onChange={(event) => setSort(event.target.value as SortMode)}>
                <option value="created">Created</option>
                <option value="score">Score</option>
                <option value="name">Name</option>
              </select>
            </div>
          </div>

          <div className={p.modelAnalysisScrollV31}>
            <QueryState
              size="block"
              isEmpty={!models.length}
              emptyTitle="No models yet"
              emptyDescription="Train or import a model to start comparing metrics."
            >
              <MetricsTodoBanner />

              <GraphSection title="Metrics" icon={BarChart3} count={`${metricCharts.length}/${metricCharts.length}`} defaultOpen>
                <div className={p.curveGridV31}>
                  {metricCharts.map((chart) => (
                    <CurveCard key={chart.key} chart={chart} models={chartModels} tasks={tasks} />
                  ))}
                </div>
              </GraphSection>

              <GraphSection title="Loss" icon={Activity} count={`${lossCharts.length}/${lossCharts.length}`}>
                <div className={p.curveGridV31}>
                  {lossCharts.map((chart) => (
                    <CurveCard key={chart.key} chart={chart} models={chartModels} tasks={tasks} />
                  ))}
                </div>
              </GraphSection>

              <GraphSection title="Learning Rate" icon={RefreshCw} count={`${lrCharts.length}/${lrCharts.length}`}>
                <div className={p.curveGridV31}>
                  {lrCharts.map((chart) => (
                    <CurveCard key={chart.key} chart={chart} models={chartModels} tasks={tasks} />
                  ))}
                </div>
              </GraphSection>

              <GraphSection title="Dataset & model metadata" icon={Database} count={selectedModel ? modelTitle(selectedModel) : "—"} defaultOpen>
                <ModelMetadataPanel
                  groupId={group.id}
                  groupStats={group.stats}
                  model={selectedModel}
                  task={selectedModel ? getModelTask(selectedModel, tasks) : null}
                  onActivate={() => selectedModel && activateMutation.mutate(selectedModel.id)}
                  onDelete={() => {
                    if (!selectedModel) return;
                    if (!window.confirm("Удалить модель? Это действие нельзя отменить.")) return;
                    deleteMutation.mutate(selectedModel.id, {
                      onSuccess: () => navigate(paths.trainingModels(group.id)),
                    });
                  }}
                  onPause={(taskId) => pauseMutation.mutate(taskId)}
                  onResume={(taskId) => resumeMutation.mutate(taskId)}
                  onCancel={(taskId) => cancelMutation.mutate(taskId)}
                  isActivating={Boolean(selectedModel && activateMutation.isPending && activateMutation.variables === selectedModel.id)}
                  isDeleting={Boolean(selectedModel && deleteMutation.isPending && deleteMutation.variables === selectedModel.id)}
                  isPausing={pauseMutation.isPending}
                  isResuming={resumeMutation.isPending}
                  isCancelling={cancelMutation.isPending}
                />
              </GraphSection>
            </QueryState>
          </div>
        </main>
      </section>
    </div>
  );
}

function MetricsTodoBanner() {
  return (
    <article className={p.metricsTodoBannerV31}>
      <AlertTriangle />
      <div>
        <strong>TODO backend: автоматическое извлечение метрик</strong>
        <span>
          Сейчас UI умеет читать history/curves, если backend положит их в model/task. Для поведения как у платформы нужно на import/train парсить checkpoint artifacts/results.csv и сохранять per-epoch curves в модель.
        </span>
      </div>
    </article>
  );
}

function GraphSection({ title, icon: Icon, count, defaultOpen = false, children }: { title: string; icon: LucideIcon; count: string; defaultOpen?: boolean; children: ReactNode }) {
  return (
    <details className={p.graphSectionV31} open={defaultOpen}>
      <summary>
        <span><Icon /> {title}</span>
        <b>{count}</b>
      </summary>
      <div className={p.graphSectionBodyV31}>{children}</div>
    </details>
  );
}

function CurveCard({ chart, models, tasks }: { chart: ChartDefinition; models: MlModel[]; tasks: TaskLike[] }) {
  const series = models.map((model) => seriesForChart(model, getModelTask(model, tasks), chart));
  const nonEmpty = series.filter((item) => item.values.length > 1);
  const allValues = nonEmpty.flatMap((item) => item.values);
  const min = allValues.length ? Math.min(...allValues) : 0;
  const max = allValues.length ? Math.max(...allValues) : 1;
  const top = allValues.length ? allValues[allValues.length - 1] : null;
  const sources = Array.from(new Set(series.map((item) => item.source)));
  const sourceLabel = sources.includes("history") ? "history" : sources.includes("summary") ? "summary-derived" : "todo backend";

  return (
    <article className={p.curveCardV31}>
      <header>
        <div>
          <span>{chart.title}</span>
          <b>{formatChartValue(top, chart.kind)}</b>
        </div>
        <em>{sourceLabel}</em>
      </header>

      <div className={p.curvePlotV31}>
        {nonEmpty.length ? (
          <svg viewBox="0 0 320 150" preserveAspectRatio="none" role="img" aria-label={chart.title}>
            <g className={p.curveGridLinesV31}>
              <line x1="0" x2="320" y1="37" y2="37" />
              <line x1="0" x2="320" y1="75" y2="75" />
              <line x1="0" x2="320" y1="113" y2="113" />
            </g>
            {nonEmpty.map((item, index) => (
              <polyline
                key={item.model.id}
                className={p.curveLineV31}
                style={{ "--series-index": index } as CSSProperties}
                points={pointsForSeries(item.values, min, max)}
              />
            ))}
          </svg>
        ) : (
          <div className={p.curveTodoV31}>
            <FileJson />
            <strong>No curve data</strong>
            <span>Нужен backend parser для results.csv/checkpoint metadata.</span>
          </div>
        )}
      </div>

      <footer>
        {series.map((item, index) => (
          <span key={item.model.id} style={{ "--series-index": index } as CSSProperties}>
            <i /> {modelTitle(item.model)}
          </span>
        ))}
      </footer>
    </article>
  );
}

function pointsForSeries(values: number[], min: number, max: number) {
  const width = 320;
  const height = 150;
  const range = Math.max(0.000001, max - min);
  return values
    .map((value, index) => {
      const x = values.length === 1 ? width : (index / (values.length - 1)) * width;
      const y = height - ((value - min) / range) * (height - 14) - 7;
      return `${Math.round(x * 10) / 10},${Math.round(y * 10) / 10}`;
    })
    .join(" ");
}

function ModelMetadataPanel({
  groupId,
  groupStats,
  model,
  task,
  onActivate,
  onDelete,
  onPause,
  onResume,
  onCancel,
  isActivating,
  isDeleting,
  isPausing,
  isResuming,
  isCancelling,
}: {
  groupId: string;
  groupStats: { standards_count: number; images_count: number; annotated_images_count: number; segment_classes_count: number; polygons_count: number };
  model: MlModel | null;
  task: TaskLike | null;
  onActivate: () => void;
  onDelete: () => void;
  onPause: (taskId: string) => void;
  onResume: (taskId: string) => void;
  onCancel: (taskId: string) => void;
  isActivating: boolean;
  isDeleting: boolean;
  isPausing: boolean;
  isResuming: boolean;
  isCancelling: boolean;
}) {
  if (!model) {
    return <div className={p.trainEmptyV27}><CircleDashed /><strong>Select a model</strong><span>Открой модель слева или запусти обучение.</span></div>;
  }

  const progress = task?.progress_percent ?? (model.trained_at ? 100 : 0);
  const classCount = model.class_meta?.length ?? model.class_keys?.length ?? model.num_classes ?? 0;
  const labeledPercent = percent(groupStats.annotated_images_count, groupStats.images_count);
  const isActive = task ? isActiveTaskStatus(task.status) : false;
  const isPaused = task?.status === "paused";

  return (
    <div className={p.modelMetadataGridV31}>
      <section className={p.modelSummaryCardV31}>
        <div className={p.modelInspectorHeroV31}>
          <div className={p.trainModelIconV27}><Brain /></div>
          <div>
            <span className={p.eyebrow}><ShieldCheck /> Selected model</span>
            <h2>{modelTitle(model)}</h2>
            <p>{model.is_active ? "Active model" : model.trained_at ? "Ready model" : "Draft model"} · created {formatDate(model.created_at)}</p>
          </div>
          <div className={p.trainTaskActionsV27}>
            {!model.is_active ? <button type="button" onClick={onActivate} disabled={isActivating}><CheckCircle2 /> Activate</button> : null}
            <button type="button" onClick={onDelete} disabled={isDeleting}><Trash2 /> Delete</button>
          </div>
        </div>

        <div className={p.trainMetricMiniGridV27}>
          {metricNames.map((name) => <small key={name}><b>{name}</b><span>{formatMetric(model.metrics?.[name])}</span></small>)}
        </div>
      </section>

      <section className={p.modelStatusCardV31}>
        <div className={p.trainPanelHeaderV27}>
          <div>
            <span className={p.eyebrow}>Training status</span>
            <h2>{task ? task.type : model.trained_at ? "trained" : "no job"}</h2>
            <p>{task ? `${task.status} · ${task.stage ?? "queued"}` : model.trained_at ? `trained ${formatDate(model.trained_at)}` : "No linked training job"}</p>
          </div>
          <strong>{progress}%</strong>
        </div>
        <div className={p.trainProgressV27}><span style={{ width: `${progress}%` }} /></div>
        <small>{task?.message ?? task?.error ?? "Training/import/export status is stored here, not in a separate side rail."}</small>
        {task ? (
          <div className={p.trainTaskActionsV27}>
            {isActive ? <button type="button" onClick={() => onPause(task.id)} disabled={isPausing}><Pause /> Pause</button> : null}
            {isPaused ? <button type="button" onClick={() => onResume(task.id)} disabled={isResuming}><Play /> Resume</button> : null}
            {(isActive || isPaused) ? <button type="button" onClick={() => onCancel(task.id)} disabled={isCancelling}><Trash2 /> Cancel</button> : null}
          </div>
        ) : null}
      </section>

      <section className={p.modelDatasetCardV31}>
        <h3><Database /> Dataset snapshot</h3>
        <div className={p.trainSpecGridV27}>
          <Spec icon={Image} label="References" value={groupStats.standards_count} />
          <Spec icon={Image} label="Images" value={groupStats.images_count} />
          <Spec icon={Tags} label="Labeled" value={`${labeledPercent}%`} />
          <Spec icon={Layers3} label="Classes" value={groupStats.segment_classes_count} />
          <Spec icon={GitBranch} label="Polygons" value={groupStats.polygons_count} />
          <Spec icon={Database} label="Model images" value={totalImages(model) || "—"} />
        </div>
        <Link to={paths.assetReferences(groupId)}>Open Assets references</Link>
      </section>

      <section className={p.modelDatasetCardV31}>
        <h3><Ruler /> Training params</h3>
        <div className={p.trainSpecGridV27}>
          <Spec icon={Ruler} label="Image size" value={model.imgsz ? `${model.imgsz}px` : "—"} />
          <Spec icon={Layers3} label="Classes" value={model.num_classes ?? classCount ?? "—"} />
          <Spec icon={CalendarClock} label="Epochs" value={model.epochs ?? "—"} />
          <Spec icon={Boxes} label="Batch" value={model.batch_size ?? "—"} />
          <Spec icon={Image} label="Train" value={model.train_count ?? "—"} />
          <Spec icon={Image} label="Val/Test" value={`${model.val_count ?? "—"} / ${model.test_count ?? "—"}`} />
        </div>
      </section>

      <section className={p.modelDatasetCardV31}>
        <h3><GitBranch /> Class map</h3>
        {model.class_meta?.length ? (
          <div className={p.trainClassCloudV27}>
            {model.class_meta.map((item) => (
              <span key={`${item.id}-${item.index}`}><GitBranch /> {item.name ?? item.key ?? item.native_key}</span>
            ))}
          </div>
        ) : model.class_keys?.length ? (
          <div className={p.trainClassCloudV27}>{model.class_keys.map((name) => <span key={name}><GitBranch /> {name}</span>)}</div>
        ) : (
          <div className={p.emptyLineV28}>Model class map is empty.</div>
        )}
      </section>
    </div>
  );
}

function Spec({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string | number }) {
  return (
    <div className={p.trainSpecV27}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
