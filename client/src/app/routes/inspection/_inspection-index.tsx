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
    title: "Фото",
    hint: "Проверка загруженного изображения.",
    icon: UploadCloud,
  },
  snapshot: {
    title: "Snapshot",
    hint: "Один кадр с камеры.",
    icon: Camera,
  },
  realtime: {
    title: "Realtime",
    hint: "Проверка live-потока.",
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
          <h1>Проверка изделия</h1>
          <p>Выбери режим и изделие.</p>
        </div>
        <div className={s.heroStats}>
          <MetricCard className={s.metric} icon={ShieldCheck} value={readyGroups} label="готово" variant="valueFirst" />
          <MetricCard className={s.metric} icon={Image} value={totalReferences} label="эталоны" variant="valueFirst" />
          <MetricCard className={s.metric} icon={ListChecks} value={totalRuns} label="проверки" variant="valueFirst" />
        </div>
      </section>

      <section className={s.modeGrid} aria-label="Режимы проверки">
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
            <span className={s.eyebrow}><FileImage /> Изделия</span>
            <h2>Выбери изделие</h2>
          </div>
          <span className={s.sectionCount}>{groups.length}</span>
        </div>

        <QueryState
          isLoading={isPending}
          isError={isError}
          isEmpty={!isPending && !isError && groups.length === 0}
          size="block"
          loadingText="Загружаем проекты"
          errorTitle="Не удалось загрузить проекты"
          emptyTitle="Нет проектов"
          emptyDescription="Создай изделие и добавь эталон в Assets."
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
  if (!group.stats.standards_count) return "Нет эталона";
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
      <p>{group.description || "Изделие для проверки."}</p>
      <div className={s.cardMeta}>
        <span>{group.stats.standards_count} эталонов</span>
        <span>{group.stats.segment_classes_count} классов</span>
        <span>{group.stats.polygons_count} полигонов</span>
        <span>{group.stats.inspections_count} проверок</span>
      </div>
    </Link>
  );
}
