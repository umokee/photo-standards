import { inspectionModePaths, paths, type InspectionModePath } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import type { GroupListItem } from "@/types/contracts";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Image,
  ListChecks,
  ShieldCheck,
  UploadCloud,
  Video,
  type LucideIcon,
} from "lucide-react";
import { Link, useParams } from "react-router-dom";
import p from "../platform-pages.module.scss";

function normalizeMode(value: string | undefined): InspectionModePath {
  return inspectionModePaths.includes(value as InspectionModePath)
    ? (value as InspectionModePath)
    : "photo";
}

const modeMeta: Record<InspectionModePath, { title: string; hint: string; icon: LucideIcon }> = {
  photo: {
    title: "Photo check",
    hint: "Загрузка одного изображения и запуск проверки по выбранному reference.",
    icon: UploadCloud,
  },
  snapshot: {
    title: "Camera snapshot",
    hint: "Один кадр с камеры, затем обычная проверка и сохранение результата.",
    icon: Camera,
  },
  realtime: {
    title: "Realtime station",
    hint: "Поток с камеры и живой статус без ручной загрузки файлов.",
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
    <div className={p.inspectStationPageV32}>
      <section className={p.inspectHeroV32}>
        <div>
          <span className={p.eyebrowV27}>Inspect station</span>
          <h1>Проверка изделия</h1>
          <p>
            Отдельная рабочая зона для запуска контроля: выбери режим, изделие, reference,
            источник изображения и классы деталей.
          </p>
        </div>
        <div className={p.inspectHeroStatsV32}>
          <Metric icon={ShieldCheck} value={readyGroups} label="ready projects" />
          <Metric icon={Image} value={totalReferences} label="references" />
          <Metric icon={ListChecks} value={totalRuns} label="runs" />
        </div>
      </section>

      <section className={p.inspectModeGridV32}>
        {inspectionModePaths.map((modeItem) => {
          const meta = modeMeta[modeItem];
          const Icon = meta.icon;
          return (
            <Link
              key={modeItem}
              to={paths.inspectionMode(modeItem)}
              className={modeItem === currentMode ? p.inspectModeCardActiveV32 : p.inspectModeCardV32}
            >
              <Icon />
              <strong>{meta.title}</strong>
              <span>{meta.hint}</span>
            </Link>
          );
        })}
      </section>

      <section className={p.inspectQueueShellV32}>
        <div className={p.sectionHeaderV27}>
          <div>
            <span className={p.eyebrowV27}>Projects</span>
            <h2>Выбери изделие для проверки</h2>
          </div>
          <span>{groups.length} projects</span>
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
          <div className={p.inspectProjectQueueV32}>
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
    <Link className={p.inspectProjectCardV32} to={paths.inspectionGroup(mode, group.id)}>
      <div className={p.inspectProjectAvatarV32}>{group.name.slice(0, 1).toUpperCase()}</div>
      <div className={p.inspectProjectMainV32}>
        <div className={p.inspectProjectTitleV32}>
          <strong>{group.name}</strong>
          <span data-ready={ready}>{ready ? <CheckCircle2 /> : <AlertTriangle />} {readinessLabel(group)}</span>
        </div>
        <p>{group.description || "Project inspection scope"}</p>
        <div className={p.inspectProjectMetaV32}>
          <span>{group.stats.standards_count} references</span>
          <span>{group.stats.segment_classes_count} classes</span>
          <span>{group.stats.polygons_count} polygons</span>
          <span>{group.stats.inspections_count} runs</span>
        </div>
      </div>
    </Link>
  );
}

function Metric({ icon: Icon, value, label }: { icon: LucideIcon; value: number; label: string }) {
  return (
    <div className={p.inspectHeroMetricV32}>
      <Icon />
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}
