import { inspectionModePaths, paths, type InspectionModePath } from "@/app/paths";
import { MetricCard } from "@/components/ui/metric-card/metric-card";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import type { GroupListItem } from "@/types/contracts";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  FileImage,
  Image,
  ListChecks,
  ShieldCheck,
  UploadCloud,
  Video,
  type LucideIcon,
} from "lucide-react";
import { Link, useParams } from "react-router-dom";
import s from "./_inspect-strict.module.scss";

function normalizeMode(value: string | undefined): InspectionModePath {
  return inspectionModePaths.includes(value as InspectionModePath)
    ? (value as InspectionModePath)
    : "photo";
}

const modeMeta: Record<InspectionModePath, { title: string; hint: string; icon: LucideIcon }> = {
  photo: {
    title: "Photo check",
    hint: "Один файл изображения, проверка по выбранному reference и сохранение результата.",
    icon: UploadCloud,
  },
  snapshot: {
    title: "Camera snapshot",
    hint: "Один кадр с выбранной камеры, затем обычная проверка изделия.",
    icon: Camera,
  },
  realtime: {
    title: "Realtime station",
    hint: "Live-поток с камеры и обновление результата без ручной загрузки файлов.",
    icon: Video,
  },
};

export function Component() {
  const { mode } = useParams();
  const currentMode = normalizeMode(mode);
  const { data: groups = [], isPending, isError } = useGetGroups();

  const totalReferences = groups.reduce((sum, group) => sum + group.stats.standards_count, 0);
  const totalRuns = groups.reduce((sum, group) => sum + group.stats.inspections_count, 0);
  const readyGroups = groups.filter(isReadyForInspect).length;

  return (
    <div className={s.page} data-page="index">
      <section className={s.hero}>
        <div className={s.heroMain}>
          <span className={s.eyebrow}><ShieldCheck /> Inspect station</span>
          <h1>Проверка изделия</h1>
          <p>
            Сначала выбирается режим и изделие, затем reference. На station-экране остаётся только source,
            классы и запуск проверки.
          </p>
        </div>
        <div className={s.heroStats}>
          <MetricCard className={s.metric} icon={ShieldCheck} value={readyGroups} label="ready projects" variant="valueFirst" />
          <MetricCard className={s.metric} icon={Image} value={totalReferences} label="references" variant="valueFirst" />
          <MetricCard className={s.metric} icon={ListChecks} value={totalRuns} label="runs" variant="valueFirst" />
        </div>
      </section>

      <section className={s.modeGrid} aria-label="Inspection modes">
        {inspectionModePaths.map((modeItem) => {
          const meta = modeMeta[modeItem];
          const Icon = meta.icon;
          return (
            <Link
              key={modeItem}
              to={paths.inspectionMode(modeItem)}
              className={s.modeCard}
              data-active={modeItem === currentMode}
            >
              <Icon />
              <strong>{meta.title}</strong>
              <span>{meta.hint}</span>
            </Link>
          );
        })}
      </section>

      <section className={s.queueShell}>
        <div className={s.sectionHeader}>
          <div>
            <span className={s.eyebrow}><FileImage /> Projects</span>
            <h2>Выбери изделие</h2>
          </div>
          <span className={s.sectionCount}>{groups.length} projects</span>
        </div>

        <QueryState
          isLoading={isPending}
          isError={isError}
          isEmpty={!isPending && !isError && groups.length === 0}
          size="block"
          loadingText="Загружаем проекты"
          errorTitle="Не удалось загрузить проекты"
          emptyTitle="Нет проектов"
          emptyDescription="Сначала создай изделие и добавь reference в Assets."
        >
          <div className={s.projectGrid}>
            {groups.map((group) => (
              <ProjectCard key={group.id} group={group} mode={currentMode} />
            ))}
          </div>
        </QueryState>
      </section>
    </div>
  );
}

function isReadyForInspect(group: GroupListItem) {
  return (
    group.stats.standards_count > 0 &&
    group.stats.segment_classes_count > 0 &&
    group.stats.polygons_count > 0
  );
}

function readinessLabel(group: GroupListItem) {
  if (!group.stats.standards_count) return "Нет reference";
  if (!group.stats.segment_classes_count) return "Нет классов";
  if (!group.stats.polygons_count) return "Нет разметки";
  return "Можно проверять";
}

function ProjectCard({ group, mode }: { group: GroupListItem; mode: InspectionModePath }) {
  const ready = isReadyForInspect(group);

  return (
    <Link className={s.projectCard} to={paths.inspectionGroup(mode, group.id)}>
      <div className={s.projectTop}>
        <div className={s.projectAvatar}>{group.name.slice(0, 1).toUpperCase()}</div>
        <div className={s.projectTitle}>
          <strong>{group.name}</strong>
          <span className={ready ? s.statusPillGood : s.statusPillBad}>
            {ready ? <CheckCircle2 /> : <AlertTriangle />} {readinessLabel(group)}
          </span>
        </div>
      </div>
      <p>{group.description || "Inspection scope for this product."}</p>
      <div className={s.cardMeta}>
        <span>{group.stats.standards_count} references</span>
        <span>{group.stats.segment_classes_count} classes</span>
        <span>{group.stats.polygons_count} polygons</span>
        <span>{group.stats.inspections_count} runs</span>
      </div>
    </Link>
  );
}
