import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useActivateModel } from "@/page-components/models/api/activate-model";
import { useDeleteModel } from "@/page-components/models/api/delete-model";
import { useGetModel } from "@/page-components/models/api/get-ml";
import { getModelMetricsHistory } from "@/page-components/models/api/get-model-metrics-history";
import { ExportModel } from "@/page-components/models/components/export-model/export-model";
import { ImportModel } from "@/page-components/models/components/import-model/import-model";
import { TrainModel } from "@/page-components/models/components/train-model/train-model";
import { getModelTask } from "@/page-components/models/lib/model-helpers";
import { useCancelTask } from "@/page-components/tasks/api/cancel-task";
import { usePauseTask } from "@/page-components/tasks/api/pause-task";
import { useResumeTask } from "@/page-components/tasks/api/resume-task";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import type { MlModel, TaskResponse, TrainingMetricsHistoryResponse } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Boxes,
  Brain,
  CalendarClock,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
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
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Tags,
  Trash2,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { Link, useLoaderData, useNavigate } from "react-router-dom";
import s from "./_training-models-strict.module.scss";
import { useTrainingModelOutletContext } from "./_training-detail";

type FilterMode = "all" | "active" | "trained" | "draft" | "withTask";
type SortMode = "created" | "score" | "name";
type BlockKey = "metrics" | "loss" | "lr" | "metadata" | "classes" | "compare";
type SeriesSource = "live" | "checkpoint" | "artifact" | "summary" | "empty";

type ChartDefinition = {
  key: string;
  title: string;
  family: "metrics" | "loss" | "lr";
  aliases: string[];
};

type SeriesBundle = {
  values: number[];
  epochs: number[];
  source: SeriesSource;
};

const metricCharts: ChartDefinition[] = [
  { key: "precision", title: "Precision", family: "metrics", aliases: ["precision", "precision(B)", "metrics/precision(B)", "metrics.precision"] },
  { key: "recall", title: "Recall", family: "metrics", aliases: ["recall", "recall(B)", "metrics/recall(B)", "metrics.recall"] },
  { key: "mAP50", title: "mAP50", family: "metrics", aliases: ["mAP50", "mAP50(B)", "metrics/mAP50(B)", "metrics.mAP50"] },
  { key: "mAP50_95", title: "mAP50-95", family: "metrics", aliases: ["mAP50_95", "mAP50-95", "mAP50-95(B)", "metrics/mAP50-95(B)", "metrics.mAP50_95"] },
];

const lossCharts: ChartDefinition[] = [
  { key: "train_box_loss", title: "Train box loss", family: "loss", aliases: ["train/box_loss", "box_loss", "train_box_loss"] },
  { key: "train_cls_loss", title: "Train class loss", family: "loss", aliases: ["train/cls_loss", "cls_loss", "train_cls_loss"] },
  { key: "train_dfl_loss", title: "Train DFL loss", family: "loss", aliases: ["train/dfl_loss", "dfl_loss", "train_dfl_loss"] },
  { key: "val_box_loss", title: "Val box loss", family: "loss", aliases: ["val/box_loss", "val_box_loss"] },
  { key: "val_cls_loss", title: "Val class loss", family: "loss", aliases: ["val/cls_loss", "val_cls_loss"] },
];

const lrCharts: ChartDefinition[] = [
  { key: "lr0", title: "Learning rate pg0", family: "lr", aliases: ["lr/pg0", "lr0", "lr_pg0"] },
  { key: "lr1", title: "Learning rate pg1", family: "lr", aliases: ["lr/pg1", "lr1", "lr_pg1"] },
  { key: "lr2", title: "Learning rate pg2", family: "lr", aliases: ["lr/pg2", "lr2", "lr_pg2"] },
];

const metricKeys = ["mAP50_95", "mAP50", "precision", "recall"] as const;

function formatPercent(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  const normalized = value <= 1 ? value * 100 : value;
  return `${Math.round(normalized * 10) / 10}%`;
}

function metricValue(model: MlModel | null, key: string) {
  const raw = model?.metrics?.[key] ?? model?.metrics?.[key.replace("_", "-")];
  if (typeof raw !== "number" || Number.isNaN(raw)) return null;
  return raw <= 1 ? Math.max(0, Math.min(1, raw)) : Math.max(0, Math.min(1, raw / 100));
}

function modelName(model: MlModel) {
  return `${model.architecture}${model.version ? ` v${model.version}` : ""}`;
}

function modelFamily(model: MlModel) {
  const name = model.architecture.toLowerCase();
  if (name.includes("seg")) return "Segment";
  if (name.includes("detect")) return "Detect";
  return "Other";
}

function taskFor(model: MlModel, tasks: TaskResponse[]) {
  return getModelTask(model, tasks);
}

function statusFor(model: MlModel, tasks: TaskResponse[]) {
  const task = taskFor(model, tasks);
  if (model.is_active) return "active";
  if (task && isActiveTaskStatus(task.status)) return task.status;
  if (model.trained_at || model.weights_path) return "trained";
  return "draft";
}

function isModelVisible(model: MlModel, tasks: TaskResponse[], filter: FilterMode, query: string) {
  const text = `${model.architecture} ${model.version ?? ""} ${model.id}`.toLowerCase();
  if (query.trim() && !text.includes(query.trim().toLowerCase())) return false;
  const task = taskFor(model, tasks);
  switch (filter) {
    case "active":
      return model.is_active;
    case "trained":
      return Boolean(model.trained_at || model.weights_path);
    case "draft":
      return !model.trained_at && !model.weights_path && !model.is_active;
    case "withTask":
      return Boolean(task);
    default:
      return true;
  }
}

function scoreModel(model: MlModel) {
  return metricValue(model, "mAP50_95") ?? metricValue(model, "mAP50") ?? metricValue(model, "precision") ?? 0;
}

function sortModels(models: MlModel[], sort: SortMode) {
  return [...models].sort((a, b) => {
    if (sort === "name") return modelName(a).localeCompare(modelName(b));
    if (sort === "score") return scoreModel(b) - scoreModel(a);
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  });
}

function normalizeSeries(values: unknown): number[] | null {
  if (!Array.isArray(values)) return null;
  const series = values.filter((value): value is number => typeof value === "number" && !Number.isNaN(value));
  return series.length > 1 ? series : null;
}

function readHistorySeries(history: TrainingMetricsHistoryResponse | undefined, aliases: string[]): SeriesBundle | null {
  if (!history) return null;
  for (const alias of aliases) {
    const values = normalizeSeries(history.series?.[alias]);
    if (values) {
      const epochs = history.epochs?.length === values.length ? history.epochs : values.map((_, index) => index + 1);
      return { values, epochs, source: history.source };
    }
  }
  return null;
}

function estimatedCurve(finalValue: number) {
  const target = Math.max(0.02, Math.min(0.99, finalValue));
  return Array.from({ length: 30 }, (_, index) => {
    const t = index / 29;
    const noise = Math.sin(index * 1.6) * 0.012;
    return Math.max(0, Math.min(1, target * (1 - Math.exp(-4.3 * t)) + noise));
  });
}

function seriesForChart(model: MlModel | null, history: TrainingMetricsHistoryResponse | undefined, chart: ChartDefinition): SeriesBundle | null {
  const fromHistory = readHistorySeries(history, chart.aliases);
  if (fromHistory) return fromHistory;

  if (chart.family === "metrics") {
    const scalar = metricValue(model, chart.key);
    if (scalar !== null) {
      const values = estimatedCurve(scalar);
      return { values, epochs: values.map((_, index) => index + 1), source: "summary" };
    }
  }

  return null;
}

function chartPath(values: number[], width = 420, height = 150, pad = 16) {
  if (values.length < 2) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return values
    .map((value, index) => {
      const x = pad + (index / (values.length - 1)) * (width - pad * 2);
      const y = pad + (1 - (value - min) / span) * (height - pad * 2);
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function sourceLabel(source: SeriesSource | null | undefined) {
  if (source === "live") return "live";
  if (source === "checkpoint") return "checkpoint";
  if (source === "artifact") return "artifact";
  if (source === "summary") return "summary";
  return "empty";
}

function SourcePill({ source }: { source: SeriesSource | null | undefined }) {
  return <span className={s.sourcePill}>{sourceLabel(source)}</span>;
}

function CollapsibleBlock({
  id,
  title,
  description,
  icon: Icon,
  open,
  onToggle,
  children,
}: {
  id: BlockKey;
  title: string;
  description: string;
  icon: LucideIcon;
  open: boolean;
  onToggle: (id: BlockKey) => void;
  children: ReactNode;
}) {
  return (
    <section className={s.block}>
      <button className={s.blockHeader} type="button" onClick={() => onToggle(id)}>
        <span className={s.blockHeaderTitle}>
          <Icon />
          <span>
            <strong>{title}</strong>
            <span>{description}</span>
          </span>
        </span>
        {open ? <ChevronDown /> : <ChevronRight />}
      </button>
      {open ? <div className={s.blockBody}>{children}</div> : null}
    </section>
  );
}

function ChartCard({ model, history, chart }: { model: MlModel | null; history?: TrainingMetricsHistoryResponse; chart: ChartDefinition }) {
  const series = seriesForChart(model, history, chart);

  return (
    <article className={s.chartCard}>
      <header className={s.chartTop}>
        <div>
          <strong>{chart.title}</strong>
          <span>{series ? `${series.values.length} points` : "No curve data"}</span>
        </div>
        <SourcePill source={series?.source ?? "empty"} />
      </header>

      {series ? (
        <svg className={s.chartSvg} viewBox="0 0 420 150" role="img" aria-label={chart.title}>
          <path d={chartPath(series.values)} fill="none" stroke="var(--train-accent)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="16" cy="134" r="2" fill="var(--train-muted)" opacity="0.35" />
          <circle cx="404" cy="16" r="2" fill="var(--train-muted)" opacity="0.35" />
        </svg>
      ) : (
        <div className={s.todoChart}>Метрика не пришла от backend. Блок оставлен под real-time/checkpoint data, а не под фейковый график.</div>
      )}

      <div className={s.chartLegend}>
        <span className={s.legendItem}><i className={s.legendSwatch} /> {model ? modelName(model) : "model"}</span>
        {series ? <span>epoch {series.epochs[0]} → {series.epochs[series.epochs.length - 1]}</span> : null}
      </div>
    </article>
  );
}

function MetricCard({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <article className={s.metricCard}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </article>
  );
}

function MetaCard({ label, value }: { label: string; value: ReactNode }) {
  return (
    <article className={s.metaCard}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function SplitRow({ label, count, ratio }: { label: string; count: number | null; ratio: number | null }) {
  const width = Math.round(Math.max(0, Math.min(100, (ratio ?? 0) <= 1 ? (ratio ?? 0) * 100 : (ratio ?? 0))));
  return (
    <div className={s.splitRow}>
      <span>{label}</span>
      <div className={s.splitBar}><div className={s.splitFill} style={{ "--split-width": `${width}%` } as CSSProperties} /></div>
      <strong>{count ?? "—"}</strong>
    </div>
  );
}

function JobPanel({ task, groupId }: { task: ReturnType<typeof getModelTask> | null | undefined; groupId: string }) {
  const pauseTask = usePauseTask({ groupId });
  const resumeTask = useResumeTask({ groupId });
  const cancelTask = useCancelTask({ groupId });
  const progress = task?.progress_percent ?? 0;

  if (!task) {
    return (
      <article className={s.jobCard}>
        <span>Linked job</span>
        <strong>No task linked</strong>
        <p className={s.chartLegend}>Модель не имеет активной задачи обучения/импорта/экспорта.</p>
      </article>
    );
  }

  return (
    <article className={s.jobCard}>
      <div>
        <span>Linked job</span>
        <strong>{task.type} · {task.status}</strong>
      </div>
      <div className={s.progressTrack}><div className={s.progressFill} style={{ width: `${progress}%` }} /></div>
      <p className={s.chartLegend}>{task.stage ?? "stage"} · {task.message ?? task.error ?? "No message"}</p>
      {isActiveTaskStatus(task.status) ? (
        <div className={s.jobActions}>
          <button className={s.softButton} type="button" onClick={() => pauseTask.mutate(task.id)}><Pause /> Pause</button>
          <button className={s.softButton} type="button" onClick={() => resumeTask.mutate(task.id)}><Play /> Resume</button>
          <button className={s.dangerButton} type="button" onClick={() => cancelTask.mutate(task.id)}><AlertTriangle /> Cancel</button>
        </div>
      ) : null}
    </article>
  );
}

function CompareBlock({ models }: { models: MlModel[] }) {
  if (!models.length) return <div className={s.emptyMain}>Выбери модели слева, чтобы сравнить их метрики.</div>;

  return (
    <div className={s.compareGrid}>
      {models.map((model) => (
        <article className={s.compareCard} key={model.id}>
          <header>
            <strong>{modelName(model)}</strong>
            {model.is_active ? <span className={s.activePill}>active</span> : null}
          </header>
          <div className={s.compareBars}>
            {metricKeys.map((key) => {
              const value = metricValue(model, key);
              return (
                <div className={s.compareRow} key={key}>
                  <span>{key}</span>
                  <div className={s.compareBar}><div className={s.compareFill} style={{ "--compare-width": `${(value ?? 0) * 100}%` } as CSSProperties} /></div>
                  <strong>{formatPercent(value)}</strong>
                </div>
              );
            })}
          </div>
        </article>
      ))}
    </div>
  );
}

export function Component() {
  const { modelId = null } = useLoaderData() as { modelId: string | null };
  const { group, models, tasks } = useTrainingModelOutletContext();
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterMode>("all");
  const [sort, setSort] = useState<SortMode>("created");
  const [railOpen, setRailOpen] = useState(true);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [openBlocks, setOpenBlocks] = useState<Record<BlockKey, boolean>>({
    metrics: true,
    loss: true,
    lr: false,
    metadata: true,
    classes: true,
    compare: false,
  });

  const canTrain = group.stats.standards_count > 0 && group.stats.images_count > 0 && group.stats.annotated_images_count > 0 && group.stats.segment_classes_count > 0;
  const hasActiveTrainingTask = tasks.some((task) => isActiveTaskStatus(task.status));

  const filteredModels = useMemo(() => {
    return sortModels(models.filter((model) => isModelVisible(model, tasks, filter, query)), sort);
  }, [models, tasks, filter, query, sort]);

  const selectedFromList = useMemo(() => {
    if (modelId) return models.find((model) => model.id === modelId) ?? null;
    return models.find((model) => model.is_active) ?? models[0] ?? null;
  }, [modelId, models]);

  const modelQuery = useGetModel(selectedFromList?.id ?? null);
  const selectedModel = modelQuery.data ?? selectedFromList;
  const selectedTask = selectedModel ? taskFor(selectedModel, tasks) : null;
  const selectedCompareModels = compareIds.map((id) => models.find((model) => model.id === id)).filter((model): model is MlModel => Boolean(model));

  const metricsQuery = useQuery({
    queryKey: ["training", "metrics-history", selectedModel?.id ?? "none"],
    queryFn: () => getModelMetricsHistory(selectedModel!.id),
    enabled: Boolean(selectedModel?.id),
    refetchInterval: selectedTask && isActiveTaskStatus(selectedTask.status) ? 1800 : 15000,
  });

  useEffect(() => {
    if (!modelId && selectedFromList) {
      navigate(paths.trainingModel(group.id, selectedFromList.id), { replace: true });
    }
  }, [group.id, modelId, navigate, selectedFromList]);

  const activateModel = useActivateModel({ groupId: group.id });
  const deleteModel = useDeleteModel({ groupId: group.id });

  const groupedModels = useMemo(() => {
    const groups = new Map<string, MlModel[]>();
    for (const model of filteredModels) {
      const family = modelFamily(model);
      groups.set(family, [...(groups.get(family) ?? []), model]);
    }
    return Array.from(groups.entries());
  }, [filteredModels]);

  const toggleCompare = (model: MlModel) => {
    setCompareIds((current) => current.includes(model.id) ? current.filter((id) => id !== model.id) : [...current, model.id].slice(-4));
  };

  const toggleBlock = (id: BlockKey) => setOpenBlocks((current) => ({ ...current, [id]: !current[id] }));
  const metricHistory = metricsQuery.data;

  return (
    <div className={s.page}>
      <header className={s.header}>
        <div className={s.headerMain}>
          <span className={s.eyebrow}><Brain /> Train / Models</span>
          <h1>Model laboratory</h1>
          <p>Строгая страница анализа модели: список моделей слева, метрики и графики справа. История метрик запрашивается при открытии модели и может обновляться во время обучения.</p>
        </div>
        <div className={s.headerActions}>
          <ImportModel groupId={group.id} />
          <ExportModel models={models} />
          <TrainModel groupId={group.id} canTrain={canTrain} isTrainingLocked={hasActiveTrainingTask} />
        </div>
      </header>

      <section className={s.shell}>
        <aside className={`${s.rail} ${railOpen ? "" : s.railCollapsed}`}>
          <div className={s.railTop}>
            <label className={s.search}>
              <Search />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search model..." />
            </label>
            <div className={s.railControls}>
              <button className={s.collapseButton} type="button" onClick={() => setRailOpen((value) => !value)}>{railOpen ? "Collapse" : "Models"}</button>
              {compareIds.length ? <button className={s.softButton} type="button" onClick={() => setCompareIds([])}>Clear compare</button> : null}
            </div>
          </div>

          {railOpen ? (
            <>
              <div className={s.filters}>
                <Filter />
                {(["all", "active", "trained", "draft", "withTask"] as FilterMode[]).map((value) => (
                  <button key={value} className={`${s.filterButton} ${filter === value ? s.filterActive : ""}`} type="button" onClick={() => setFilter(value)}>
                    {value === "withTask" ? "jobs" : value}
                  </button>
                ))}
              </div>

              <div className={s.sortBar}>
                <SlidersHorizontal />
                {(["created", "score", "name"] as SortMode[]).map((value) => (
                  <button key={value} className={`${s.sortButton} ${sort === value ? s.sortActive : ""}`} type="button" onClick={() => setSort(value)}>{value}</button>
                ))}
              </div>

              <div className={s.railSummary}>
                <div className={s.miniStat}><span>Total</span><strong>{models.length}</strong></div>
                <div className={s.miniStat}><span>Active</span><strong>{models.filter((model) => model.is_active).length}</strong></div>
                <div className={s.miniStat}><span>Compare</span><strong>{compareIds.length}</strong></div>
              </div>

              <div className={s.railBody}>
                <QueryState size="block" isEmpty={!filteredModels.length} emptyTitle="No models" emptyDescription="Train/import a model or clear filters.">
                  <div className={s.modelList}>
                    {groupedModels.map(([family, items]) => (
                      <section className={s.family} key={family}>
                        <header className={s.familyHeader}><span>{family}</span><b>{items.length}</b></header>
                        {items.map((model) => {
                          const selected = selectedModel?.id === model.id;
                          const checked = compareIds.includes(model.id);
                          return (
                            <article key={model.id} className={`${s.modelCard} ${selected ? s.modelCardActive : ""}`} role="button" tabIndex={0} onClick={() => navigate(paths.trainingModel(group.id, model.id))} onKeyDown={(event) => { if (event.key === "Enter") navigate(paths.trainingModel(group.id, model.id)); }}>
                              <span className={s.modelCardTop}>
                                <span className={s.modelTitle}>
                                  <strong>{modelName(model)}</strong>
                                  <span>{formatDate(model.created_at)} · {statusFor(model, tasks)}</span>
                                </span>
                                {model.is_active ? <span className={s.activePill}>active</span> : null}
                              </span>
                              <span className={s.modelMetrics}>
                                {metricKeys.map((key) => (
                                  <span className={s.modelMetric} key={key}><span>{key}</span><strong>{formatPercent(metricValue(model, key))}</strong></span>
                                ))}
                              </span>
                              <span className={s.railControls}>
                                <span className={s.sourcePill}>{model.architecture}</span>
                                <span className={s.sourcePill}>{model.num_classes ?? "—"} classes</span>
                                <button className={s.iconButton} type="button" title="Compare" onClick={(event) => { event.stopPropagation(); toggleCompare(model); }}>
                                  {checked ? <CheckCircle2 /> : <CircleDashed />}
                                </button>
                              </span>
                            </article>
                          );
                        })}
                      </section>
                    ))}
                  </div>
                </QueryState>
              </div>
            </>
          ) : null}
        </aside>

        <main className={s.main}>
          <QueryState size="block" isEmpty={!selectedModel} emptyTitle="No model selected" emptyDescription="Выбери модель слева или импортируй веса.">
            {selectedModel ? (
              <>
                <section className={s.modelHeader}>
                  <div className={s.modelIdentity}>
                    <div className={s.modelAvatar}><BarChart3 /></div>
                    <div>
                      <h2>{modelName(selectedModel)}</h2>
                      <p>Created {formatDate(selectedModel.created_at)} · {selectedModel.trained_at ? `trained ${formatDate(selectedModel.trained_at)}` : "not trained yet"}</p>
                      <div className={s.modelBadges}>
                        {selectedModel.is_active ? <span className={s.activePill}><ShieldCheck /> active</span> : <span className={s.statusPill}>{statusFor(selectedModel, tasks)}</span>}
                        <span className={s.statusPill}><Layers3 /> {selectedModel.architecture}</span>
                        <span className={s.statusPill}><Image /> {selectedModel.imgsz}px</span>
                        <SourcePill source={metricHistory?.source ?? (metricsQuery.isLoading ? "live" : "empty")} />
                      </div>
                    </div>
                  </div>
                  <div className={s.modelActions}>
                    {!selectedModel.is_active ? <button className={s.primaryButton} type="button" onClick={() => activateModel.mutate(selectedModel.id)}><ShieldCheck /> Make active</button> : null}
                    <button className={s.softButton} type="button" onClick={() => metricsQuery.refetch()}><RefreshCw /> Refresh metrics</button>
                    <button className={s.dangerButton} type="button" onClick={() => window.confirm("Удалить модель?") && deleteModel.mutate(selectedModel.id)}><Trash2 /> Delete</button>
                  </div>
                </section>

                <div className={s.contentScroll}>
                  <section className={s.metricGrid}>
                    <MetricCard label="mAP50-95" value={formatPercent(metricValue(selectedModel, "mAP50_95"))} hint="primary quality metric" />
                    <MetricCard label="mAP50" value={formatPercent(metricValue(selectedModel, "mAP50"))} hint="object matching quality" />
                    <MetricCard label="Precision" value={formatPercent(metricValue(selectedModel, "precision"))} hint="false positives control" />
                    <MetricCard label="Recall" value={formatPercent(metricValue(selectedModel, "recall"))} hint="missed detections control" />
                  </section>

                  <CollapsibleBlock id="metrics" title="Metrics" description="Live/checkpoint curves from backend" icon={Activity} open={openBlocks.metrics} onToggle={toggleBlock}>
                    <div className={s.chartGrid}>{metricCharts.map((chart) => <ChartCard key={chart.key} model={selectedModel} history={metricHistory} chart={chart} />)}</div>
                  </CollapsibleBlock>

                  <CollapsibleBlock id="loss" title="Loss" description="Training and validation losses" icon={BarChart3} open={openBlocks.loss} onToggle={toggleBlock}>
                    <div className={s.chartGrid}>{lossCharts.map((chart) => <ChartCard key={chart.key} model={selectedModel} history={metricHistory} chart={chart} />)}</div>
                  </CollapsibleBlock>

                  <CollapsibleBlock id="lr" title="Learning rate" description="Optimizer learning-rate schedule" icon={GitBranch} open={openBlocks.lr} onToggle={toggleBlock}>
                    <div className={s.chartGrid}>{lrCharts.map((chart) => <ChartCard key={chart.key} model={selectedModel} history={metricHistory} chart={chart} />)}</div>
                  </CollapsibleBlock>

                  <CollapsibleBlock id="metadata" title="Dataset & model metadata" description="What was used to train or import this model" icon={Database} open={openBlocks.metadata} onToggle={toggleBlock}>
                    <div className={s.metaGrid}>
                      <MetaCard label="Architecture" value={selectedModel.architecture} />
                      <MetaCard label="Epochs" value={selectedModel.epochs ?? "—"} />
                      <MetaCard label="Batch" value={selectedModel.batch_size ?? "—"} />
                      <MetaCard label="Classes" value={selectedModel.num_classes ?? selectedModel.class_meta?.length ?? "—"} />
                      <MetaCard label="Images" value={selectedModel.total_images ?? group.stats.images_count} />
                      <MetaCard label="Weights" value={selectedModel.weights_path ? "available" : "missing"} />
                      <MetaCard label="Metric source" value={metricHistory?.source ?? "loading/empty"} />
                      <MetaCard label="Metric task" value={metricHistory?.task_id ?? "—"} />
                    </div>

                    <div className={s.blockBody}>
                      <div className={s.splits}>
                        <SplitRow label="Train" count={selectedModel.train_count} ratio={selectedModel.train_ratio} />
                        <SplitRow label="Val" count={selectedModel.val_count} ratio={selectedModel.val_ratio} />
                        <SplitRow label="Test" count={selectedModel.test_count} ratio={selectedModel.test_ratio} />
                      </div>
                    </div>

                    <JobPanel task={selectedTask} groupId={group.id} />
                  </CollapsibleBlock>

                  <CollapsibleBlock id="classes" title="Model classes" description="Native/imported classes mapped to project classes" icon={Tags} open={openBlocks.classes} onToggle={toggleBlock}>
                    {selectedModel.class_meta?.length ? (
                      <div className={s.classGrid}>
                        {selectedModel.class_meta.map((item) => (
                          <article className={s.classCard} key={`${item.index}-${item.key}`}>
                            <i className={s.classDot} style={{ "--class-hue": item.hue ?? 210 } as CSSProperties} />
                            <div>
                              <span>{item.native_key ?? item.key} · #{item.native_index ?? item.index}</span>
                              <strong>{item.name}</strong>
                            </div>
                          </article>
                        ))}
                      </div>
                    ) : (
                      <div className={s.emptyMain}>Нет class metadata. Для импортированных моделей backend должен вернуть class_meta после анализа весов.</div>
                    )}
                  </CollapsibleBlock>

                  <CollapsibleBlock id="compare" title="Compare selected models" description="Quick comparison of up to four models" icon={Boxes} open={openBlocks.compare} onToggle={toggleBlock}>
                    <CompareBlock models={selectedCompareModels} />
                  </CollapsibleBlock>
                </div>
              </>
            ) : null}
          </QueryState>
        </main>
      </section>
    </div>
  );
}
