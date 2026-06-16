import QueryState from "@/components/ui/query-state/query-state";
import { formatDate } from "@/utils/formatDate";
import { Brain, Boxes, CalendarClock, Database, GitBranch, Image, Layers3, Ruler, ShieldCheck } from "lucide-react";
import { useLoaderData } from "react-router-dom";
import p from "../platform-pages.module.scss";
import { useTrainingModelOutletContext } from "./_training-detail";

const metricNames = ["mAP50", "mAP50_95", "precision", "recall"] as const;

function formatMetric(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "—";
  return value <= 1 ? `${Math.round(value * 100)}%` : String(Math.round(value * 100) / 100);
}

export function Component() {
  const { modelId } = useLoaderData() as { modelId: string | null };
  const { models } = useTrainingModelOutletContext();
  const model = models.find((item) => item.id === modelId) ?? null;

  if (!modelId) {
    return (
      <section className={p.modelDetailShell}>
        <div className={p.emptyFocusCard}>
          <Brain />
          <strong>Select a model</strong>
          <span>Открой версию из списка слева, чтобы увидеть метрики и состав классов.</span>
        </div>
      </section>
    );
  }

  return (
    <QueryState isEmpty={!model} size="block" emptyTitle="Model not found">
      {model ? (
        <section className={p.modelDetailShell}>
          <div className={p.modelDetailHeader}>
            <div className={p.activeModelIcon}><Brain /></div>
            <div>
              <span className={p.eyebrow}><ShieldCheck /> Model detail</span>
              <h3>{model.architecture} {model.version ? `v${model.version}` : ""}</h3>
              <p>{model.is_active ? "Active model" : "Ready model"} · created {formatDate(model.created_at)}</p>
            </div>
          </div>

          <div className={p.metricMiniGridLarge}>
            {metricNames.map((name) => <small key={name}><b>{name}</b>{formatMetric(model.metrics?.[name])}</small>)}
          </div>

          <div className={p.modelSpecGrid}>
            <Spec icon={Ruler} label="Image size" value={`${model.imgsz}px`} />
            <Spec icon={Layers3} label="Classes" value={model.num_classes ?? "—"} />
            <Spec icon={CalendarClock} label="Epochs" value={model.epochs ?? "—"} />
            <Spec icon={Boxes} label="Batch" value={model.batch_size ?? "—"} />
            <Spec icon={Image} label="Train / Val / Test" value={`${model.train_count ?? "—"} / ${model.val_count ?? "—"} / ${model.test_count ?? "—"}`} />
            <Spec icon={Database} label="Total images" value={model.total_images ?? "—"} />
          </div>

          {model.class_meta?.length ? (
            <div className={p.classChipCloud}>
              {model.class_meta.map((item) => (
                <span key={`${item.native_key ?? item.key ?? item.name}-${item.index ?? item.native_index ?? item.name}`}>
                  <GitBranch /> {item.name ?? item.key ?? item.native_key}
                </span>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}
    </QueryState>
  );
}

function Spec({ icon: Icon, label, value }: { icon: typeof Brain; label: string; value: string | number }) {
  return (
    <div className={p.modelSpecCard}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
