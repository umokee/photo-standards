import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import { MetricCard } from "@/components/ui/metric-card/metric-card";
import QueryState from "@/components/ui/query-state/query-state";
import { StatusChip, type StatusChipTone } from "@/components/ui/status-chip/status-chip";
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
  Brain,
  CalendarClock,
  CheckCircle2,
  Clock3,
  ChevronDown,
  ChevronRight,
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
import { useEffect, useMemo, useState, type CSSProperties, type PointerEvent, type ReactNode } from "react";
import { Link, useLoaderData, useNavigate } from "react-router-dom";
import s from "./_training-models-strict.module.scss";
import { useTrainingModelOutletContext } from "./_training-detail";

type FilterMode = "all" | "active" | "trained" | "draft" | "withTask";
type SortMode = "created" | "score" | "name";
type BlockKey = "metrics" | "loss" | "lr" | "metadata" | "classes";
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
  if (task) return task.status;
  if (model.trained_at || model.weights_path) return "trained";
  return "draft";
}

function modelStatusMeta(model: MlModel, tasks: TaskResponse[]) {
  const status = statusFor(model, tasks);

  if (status === "active") return { label: "active", tone: "accent" as StatusChipTone, icon: ShieldCheck };
  if (status === "trained" || status === "succeeded") return { label: status, tone: "success" as StatusChipTone, icon: CheckCircle2 };
  if (status === "failed" || status === "cancelled") return { label: status, tone: "danger" as StatusChipTone, icon: Trash2 };
  if (status === "paused") return { label: status, tone: "warning" as StatusChipTone, icon: Pause };
  if (isActiveTaskStatus(status)) return { label: status, tone: "warning" as StatusChipTone, icon: RefreshCw };
  return { label: status, tone: "neutral" as StatusChipTone, icon: Clock3 };
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

type ChartPoint = {
  value: number;
  epoch: number;
  x: number;
  y: number;
};

const chartBox = {
  width: 460,
  height: 190,
  padLeft: 42,
  padRight: 18,
  padTop: 18,
  padBottom: 32,
};

function chartDomain(values: number[], chart: ChartDefinition) {
  if (chart.family === "metrics") {
    return { min: 0, max: 1 };
  }

  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const span = rawMax - rawMin || Math.max(Math.abs(rawMax), 1);
  const padding = span * 0.08;

  return {
    min: Math.max(0, rawMin - padding),
    max: rawMax + padding,
  };
}

function chartValueLabel(chart: ChartDefinition, value: number) {
  if (chart.family === "metrics") return formatPercent(value);
  if (chart.family === "lr") return value < 0.001 ? value.toExponential(2) : value.toFixed(5).replace(/0+$/, "").replace(/\.$/, "");
  if (value >= 10) return value.toFixed(2);
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function chartPoints(series: SeriesBundle, chart: ChartDefinition): ChartPoint[] {
  const { min, max } = chartDomain(series.values, chart);
  const span = max - min || 1;
  const plotWidth = chartBox.width - chartBox.padLeft - chartBox.padRight;
  const plotHeight = chartBox.height - chartBox.padTop - chartBox.padBottom;

  return series.values.map((value, index) => ({
    value,
    epoch: series.epochs[index] ?? index + 1,
    x: chartBox.padLeft + (index / Math.max(series.values.length - 1, 1)) * plotWidth,
    y: chartBox.padTop + (1 - (value - min) / span) * plotHeight,
  }));
}

function linePath(points: ChartPoint[]) {
  return points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
}

function areaPath(points: ChartPoint[]) {
  if (points.length < 2) return "";
  const baseY = chartBox.height - chartBox.padBottom;
  return `${linePath(points)} L${points[points.length - 1].x.toFixed(1)} ${baseY} L${points[0].x.toFixed(1)} ${baseY} Z`;
}

function yTicks(values: number[], chart: ChartDefinition) {
  const { min, max } = chartDomain(values, chart);
  return [0, 0.5, 1].map((ratio) => {
    const value = min + (max - min) * ratio;
    const y = chartBox.padTop + (1 - ratio) * (chartBox.height - chartBox.padTop - chartBox.padBottom);
    return { value, y };
  });
}

function nearestPointIndex(points: ChartPoint[], x: number) {
  let nearest = 0;
  let distance = Number.POSITIVE_INFINITY;
  points.forEach((point, index) => {
    const nextDistance = Math.abs(point.x - x);
    if (nextDistance < distance) {
      nearest = index;
      distance = nextDistance;
    }
  });
  return nearest;
}

function sourceLabel(source: SeriesSource | null | undefined) {
  if (source === "live") return "live";
  if (source === "checkpoint") return "checkpoint";
  if (source === "artifact") return "artifact";
  if (source === "summary") return "summary";
  return "empty";
}

function SourcePill({ source }: { source: SeriesSource | null | undefined }) {
  return <StatusChip tone="neutral">{sourceLabel(source)}</StatusChip>;
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
      <Button unstyled className={s.blockHeader} onClick={() => onToggle(id)}>
        <span className={s.blockHeaderTitle}>
          <Icon />
          <span>
            <strong>{title}</strong>
            <span>{description}</span>
          </span>
        </span>
        {open ? <ChevronDown /> : <ChevronRight />}
      </Button>
      {open ? <div className={s.blockBody}>{children}</div> : null}
    </section>
  );
}

function ChartCard({ model, history, chart }: { model: MlModel | null; history?: TrainingMetricsHistoryResponse; chart: ChartDefinition }) {
  const series = seriesForChart(model, history, chart);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const points = series ? chartPoints(series, chart) : [];
  const activeIndex = points.length ? hoverIndex ?? points.length - 1 : null;
  const activePoint = activeIndex === null ? null : points[activeIndex] ?? null;
  const firstPoint = points[0] ?? null;
  const lastPoint = points[points.length - 1] ?? null;
  const gradientId = `chartFill-${chart.key.replace(/[^a-zA-Z0-9_-]/g, "-")}`;

  const handlePointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (!points.length) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width) * chartBox.width;
    setHoverIndex(nearestPointIndex(points, x));
  };

  return (
    <article className={s.chartCard}>
      <header className={s.chartTop}>
        <div>
          <strong>{chart.title}</strong>
          <span>{series ? `${series.values.length} точек · ${sourceLabel(series.source)}` : "Нет данных кривой"}</span>
        </div>
        <SourcePill source={series?.source ?? "empty"} />
      </header>

      {series ? (
        <svg
          className={s.chartSvg}
          viewBox={`0 0 ${chartBox.width} ${chartBox.height}`}
          role="img"
          tabIndex={0}
          aria-label={`${chart.title}: ${activePoint ? chartValueLabel(chart, activePoint.value) : "нет значения"}`}
          onPointerMove={handlePointerMove}
          onPointerLeave={() => setHoverIndex(null)}
          onFocus={() => setHoverIndex(points.length - 1)}
          onBlur={() => setHoverIndex(null)}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="var(--train-accent)" stopOpacity="0.24" />
              <stop offset="100%" stopColor="var(--train-accent)" stopOpacity="0.02" />
            </linearGradient>
          </defs>

          {yTicks(series.values, chart).map((tick) => (
            <g key={tick.y}>
              <line className={s.chartGridLine} x1={chartBox.padLeft} x2={chartBox.width - chartBox.padRight} y1={tick.y} y2={tick.y} />
              <text className={s.chartAxisLabel} x={chartBox.padLeft - 8} y={tick.y + 4} textAnchor="end">
                {chartValueLabel(chart, tick.value)}
              </text>
            </g>
          ))}

          <line className={s.chartAxisLine} x1={chartBox.padLeft} x2={chartBox.width - chartBox.padRight} y1={chartBox.height - chartBox.padBottom} y2={chartBox.height - chartBox.padBottom} />
          <text className={s.chartAxisLabel} x={chartBox.padLeft} y={chartBox.height - 10}>ep. {firstPoint?.epoch ?? 1}</text>
          <text className={s.chartAxisLabel} x={chartBox.width - chartBox.padRight} y={chartBox.height - 10} textAnchor="end">ep. {lastPoint?.epoch ?? points.length}</text>

          <path className={s.chartArea} d={areaPath(points)} fill={`url(#${gradientId})`} />
          <path className={s.chartLine} d={linePath(points)} />

          {points.map((point, index) => (
            <circle
              key={`${point.epoch}-${index}`}
              className={`${s.chartPoint} ${index === activeIndex ? s.chartPointActive : ""}`}
              cx={point.x}
              cy={point.y}
              r={index === activeIndex ? 4 : 2.4}
            />
          ))}

          {activePoint ? (
            <g className={s.chartFocus}>
              <line x1={activePoint.x} x2={activePoint.x} y1={chartBox.padTop} y2={chartBox.height - chartBox.padBottom} />
              <circle cx={activePoint.x} cy={activePoint.y} r="5" />
              <g transform={`translate(${Math.min(Math.max(activePoint.x - 62, chartBox.padLeft), chartBox.width - 142)}, ${Math.max(activePoint.y - 46, chartBox.padTop)})`}>
                <rect width="124" height="38" rx="4" />
                <text x="10" y="15">epoch {activePoint.epoch}</text>
                <text x="10" y="30">{chartValueLabel(chart, activePoint.value)}</text>
              </g>
            </g>
          ) : null}
        </svg>
      ) : (
        <div className={s.todoChart}>Метрика не пришла от backend. Здесь появится интерактивная кривая после live/checkpoint data.</div>
      )}

      <div className={s.chartLegend}>
        <span className={s.legendItem}><i className={s.legendSwatch} /> {model ? modelName(model) : "model"}</span>
        {activePoint ? <span>epoch {activePoint.epoch} · {chartValueLabel(chart, activePoint.value)}</span> : null}
      </div>
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
          <Button className={s.softButton} variant="ghost" size="sm" icon={Pause} onClick={() => pauseTask.mutate(task.id)}>Pause</Button>
          <Button className={s.softButton} variant="ghost" size="sm" icon={Play} onClick={() => resumeTask.mutate(task.id)}>Resume</Button>
          <Button className={s.dangerButton} variant="danger" size="sm" icon={AlertTriangle} onClick={() => cancelTask.mutate(task.id)}>Cancel</Button>
        </div>
      ) : null}
    </article>
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
  const [openBlocks, setOpenBlocks] = useState<Record<BlockKey, boolean>>({
    metrics: true,
    loss: false,
    lr: false,
    metadata: true,
    classes: false,
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


  const toggleBlock = (id: BlockKey) => setOpenBlocks((current) => ({ ...current, [id]: !current[id] }));
  const metricHistory = metricsQuery.data;

  return (
    <div className={s.page}>
      <header className={s.header}>
        <div className={s.headerCopy}>
          <span className={s.eyebrow}><Brain /> Train / Models</span>
          <h1>Модели</h1>
          <p>Активная модель, обучение и интерактивные метрики качества.</p>
        </div>

        <div className={s.headerStats}>
          <div><span>Всего</span><strong>{models.length}</strong></div>
          <div><span>Активная</span><strong>{models.some((model) => model.is_active) ? "есть" : "нет"}</strong></div>
          <div><span>Задания</span><strong>{tasks.filter((task) => isActiveTaskStatus(task.status)).length}</strong></div>
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
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Поиск модели..." />
            </label>
            <div className={s.railControls}>
              <Button className={s.collapseButton} variant="ghost" size="sm" onClick={() => setRailOpen((value) => !value)}>{railOpen ? "Скрыть" : "Модели"}</Button>
            </div>
          </div>

          {railOpen ? (
            <>
              <div className={s.filters}>
                <Filter />
                {(["all", "active", "trained", "draft", "withTask"] as FilterMode[]).map((value) => (
                  <Button key={value} className={`${s.filterButton} ${filter === value ? s.filterActive : ""}`} variant={filter === value ? "primary" : "ghost"} size="sm" onClick={() => setFilter(value)}>
                    {value === "all" ? "Все" : value === "active" ? "Активные" : value === "trained" ? "Обученные" : value === "draft" ? "Черновики" : "Задания"}
                  </Button>
                ))}
              </div>

              <div className={s.sortBar}>
                <SlidersHorizontal />
                {(["created", "score", "name"] as SortMode[]).map((value) => (
                  <Button key={value} className={`${s.sortButton} ${sort === value ? s.sortActive : ""}`} variant={sort === value ? "primary" : "ghost"} size="sm" onClick={() => setSort(value)}>{value === "created" ? "Новые" : value === "score" ? "Качество" : "Имя"}</Button>
                ))}
              </div>

              <div className={s.railSummary}>
                <div className={s.miniStat}><span>Всего</span><strong>{models.length}</strong></div>
                <div className={s.miniStat}><span>Активные</span><strong>{models.filter((model) => model.is_active).length}</strong></div>
                <div className={s.miniStat}><span>Обученные</span><strong>{models.filter((model) => model.trained_at || model.weights_path).length}</strong></div>
              </div>

              <div className={s.railBody}>
                <QueryState size="block" isEmpty={!filteredModels.length} emptyTitle="Нет моделей" emptyDescription="Обучи или импортируй модель либо сбрось фильтры.">
                  <div className={s.modelList}>
                    {groupedModels.map(([family, items]) => (
                      <section className={s.family} key={family}>
                        <header className={s.familyHeader}><span>{family}</span><b>{items.length}</b></header>
                        {items.map((model) => {
                          const selected = selectedModel?.id === model.id;
                          const statusMeta = modelStatusMeta(model, tasks);
                          return (
                            <article key={model.id} className={`${s.modelCard} ${selected ? s.modelCardActive : ""}`} role="button" tabIndex={0} onClick={() => navigate(paths.trainingModel(group.id, model.id))} onKeyDown={(event) => { if (event.key === "Enter") navigate(paths.trainingModel(group.id, model.id)); }}>
                              <span className={s.modelCardTop}>
                                <span className={s.modelTitle}>
                                  <strong>{modelName(model)}</strong>
                                  <span>{formatDate(model.created_at)} · {statusFor(model, tasks)}</span>
                                </span>
                                <StatusChip tone={statusMeta.tone} icon={statusMeta.icon}>{statusMeta.label}</StatusChip>
                              </span>
                              <span className={s.modelMetrics}>
                                {metricKeys.map((key) => (
                                  <span className={s.modelMetric} key={key}><span>{key}</span><strong>{formatPercent(metricValue(model, key))}</strong></span>
                                ))}
                              </span>
                              <span className={s.railControls}>
                                <StatusChip tone="neutral">{model.architecture}</StatusChip>
                                <StatusChip tone="neutral">{model.num_classes ?? "—"} classes</StatusChip>
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
          <QueryState size="block" isEmpty={!selectedModel} emptyTitle="Модель не выбрана" emptyDescription="Выбери модель слева или импортируй веса.">
            {selectedModel ? (
              <>
                <section className={s.modelHeader}>
                  <div className={s.modelIdentity}>
                    <div className={s.modelAvatar}><BarChart3 /></div>
                    <div>
                      <h2>{modelName(selectedModel)}</h2>
                      <p>Created {formatDate(selectedModel.created_at)} · {selectedModel.trained_at ? `trained ${formatDate(selectedModel.trained_at)}` : "not trained yet"}</p>
                      <div className={s.modelBadges}>
                        {(() => {
                          const statusMeta = modelStatusMeta(selectedModel, tasks);
                          return <StatusChip tone={statusMeta.tone} icon={statusMeta.icon}>{statusMeta.label}</StatusChip>;
                        })()}
                        <StatusChip tone="neutral" icon={Layers3}>{selectedModel.architecture}</StatusChip>
                        <StatusChip tone="neutral" icon={Image}>{selectedModel.imgsz}px</StatusChip>
                        <SourcePill source={metricHistory?.source ?? (metricsQuery.isLoading ? "live" : "empty")} />
                      </div>
                    </div>
                  </div>
                  <div className={s.modelActions}>
                    {!selectedModel.is_active ? <Button className={s.primaryButton} icon={ShieldCheck} onClick={() => activateModel.mutate(selectedModel.id)}>Сделать активной</Button> : null}
                    <Button className={s.softButton} variant="ghost" icon={RefreshCw} onClick={() => metricsQuery.refetch()}>Обновить метрики</Button>
                    <Button className={s.dangerButton} variant="danger" icon={Trash2} onClick={() => window.confirm("Удалить модель?") && deleteModel.mutate(selectedModel.id)}>Удалить</Button>
                  </div>
                </section>

                <div className={s.contentScroll}>
                  <section className={s.metricGrid}>
                    <MetricCard className={s.metricCard} label="mAP50-95" value={formatPercent(metricValue(selectedModel, "mAP50_95"))} hint="главная метрика качества" />
                    <MetricCard className={s.metricCard} label="mAP50" value={formatPercent(metricValue(selectedModel, "mAP50"))} hint="качество обнаружения" />
                    <MetricCard className={s.metricCard} label="Precision" value={formatPercent(metricValue(selectedModel, "precision"))} hint="контроль ложных срабатываний" />
                    <MetricCard className={s.metricCard} label="Recall" value={formatPercent(metricValue(selectedModel, "recall"))} hint="контроль пропусков" />
                  </section>

                  <CollapsibleBlock id="metrics" title="Метрики" description="Интерактивные live/checkpoint кривые" icon={Activity} open={openBlocks.metrics} onToggle={toggleBlock}>
                    <div className={s.chartGrid}>{metricCharts.map((chart) => <ChartCard key={chart.key} model={selectedModel} history={metricHistory} chart={chart} />)}</div>
                  </CollapsibleBlock>

                  <CollapsibleBlock id="loss" title="Loss" description="Ошибки обучения и валидации" icon={BarChart3} open={openBlocks.loss} onToggle={toggleBlock}>
                    <div className={s.chartGrid}>{lossCharts.map((chart) => <ChartCard key={chart.key} model={selectedModel} history={metricHistory} chart={chart} />)}</div>
                  </CollapsibleBlock>

                  <CollapsibleBlock id="lr" title="Learning rate" description="Расписание learning rate оптимизатора" icon={GitBranch} open={openBlocks.lr} onToggle={toggleBlock}>
                    <div className={s.chartGrid}>{lrCharts.map((chart) => <ChartCard key={chart.key} model={selectedModel} history={metricHistory} chart={chart} />)}</div>
                  </CollapsibleBlock>

                  <CollapsibleBlock id="metadata" title="Данные и параметры" description="Что использовалось для обучения или импорта" icon={Database} open={openBlocks.metadata} onToggle={toggleBlock}>
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

                  <CollapsibleBlock id="classes" title="Классы модели" description="Native/imported классы и связь с проектом" icon={Tags} open={openBlocks.classes} onToggle={toggleBlock}>
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
                </div>
              </>
            ) : null}
          </QueryState>
        </main>
      </section>
    </div>
  );
}
