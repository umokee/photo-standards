import { inspectionModePaths, paths, type InspectionModePath } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import type { GroupStandard } from "@/types/contracts";
import {
  AlertTriangle,
  ArrowRight,
  BadgeCheck,
  CheckCircle2,
  Clock3,
  Image,
  Layers3,
  ListChecks,
  ShieldCheck,
} from "lucide-react";
import { Link, useParams } from "react-router-dom";
import s from "./_inspect-strict.module.scss";

function normalizeMode(value: string | undefined): InspectionModePath {
  return inspectionModePaths.includes(value as InspectionModePath)
    ? (value as InspectionModePath)
    : "photo";
}

export function Component() {
  const { mode, groupId } = useParams();
  const currentMode = normalizeMode(mode);
  const { data: group, isPending, isError } = useGetGroup(groupId ?? "");

  if (!groupId) {
    return <QueryState isEmpty size="page" emptyTitle="Выбери проект" />;
  }

  return (
    <div className={s.page} data-page="group">
      <QueryState
        isLoading={isPending}
        isError={isError}
        isEmpty={!isPending && !isError && !group}
        size="block"
        loadingText="Загружаем изделие"
        errorTitle="Не удалось загрузить изделие"
        emptyTitle="Изделие не найдено"
      >
        {group ? (
          <>
            <section className={s.scopeHeader}>
              <div className={s.headerMain}>
                <span className={s.eyebrow}><ShieldCheck /> Inspect / {currentMode}</span>
                <h1>{group.name}</h1>
                <p>Выбери эталонный вид. Классы, source и запуск проверки появятся уже на station-экране.</p>
              </div>
              <div className={s.scopeActions}>
                <Link to={paths.inspectionHistoryGroup(group.id)}><ListChecks /> Runs</Link>
                <Link to={paths.assetReferences(group.id)}><Layers3 /> Assets</Link>
              </div>
            </section>

            <section className={s.readinessGrid}>
              <ReadinessCard
                ready={group.standards.length > 0}
                title="References"
                value={group.standards.length}
                hint="Нужен хотя бы один эталонный вид"
              />
              <ReadinessCard
                ready={group.stats.segment_classes_count > 0}
                title="Classes"
                value={group.stats.segment_classes_count}
                hint="Выбор деталей будет справа на station"
              />
              <ReadinessCard
                ready={group.stats.polygons_count > 0}
                title="Polygons"
                value={group.stats.polygons_count}
                hint="Ожидаемые зоны для контроля"
              />
              <ReadinessCard
                ready={Boolean(group.active_model)}
                title="Active model"
                value={group.active_model ? 1 : 0}
                hint={
                  group.active_model
                    ? `v${group.active_model.version ?? "draft"} · ${group.active_model.architecture}`
                    : "Нет активной модели"
                }
              />
            </section>

            <section className={s.referenceShell}>
              <div className={s.sectionHeader}>
                <div>
                  <span className={s.eyebrow}><Image /> References</span>
                  <h2>Выбери reference</h2>
                </div>
                <span className={s.sectionCount}>{group.standards.length} references</span>
              </div>

              <QueryState
                isEmpty={group.standards.length === 0}
                size="block"
                emptyTitle="Нет reference"
                emptyDescription="Сначала добавь reference в Assets и разметь обязательные детали."
              >
                <div className={s.referenceGrid}>
                  {group.standards.map((standard) => (
                    <ReferenceCard
                      key={standard.id}
                      standard={standard}
                      groupId={group.id}
                      mode={currentMode}
                    />
                  ))}
                </div>
              </QueryState>
            </section>
          </>
        ) : null}
      </QueryState>
    </div>
  );
}

function ReadinessCard({ ready, title, value, hint }: { ready: boolean; title: string; value: number; hint: string }) {
  return (
    <div className={s.readinessCard}>
      <div className={s.readinessTop}>
        <div className={s.readinessIcon} data-state={ready ? "ready" : "blocked"}>
          {ready ? <CheckCircle2 /> : <AlertTriangle />}
        </div>
        <span>{title}</span>
      </div>
      <strong>{value}</strong>
      <p>{hint}</p>
    </div>
  );
}

function ReferenceCard({ standard, groupId, mode }: { standard: GroupStandard; groupId: string; mode: InspectionModePath }) {
  const ready = standard.images_count > 0 && standard.annotated_images_count > 0;

  return (
    <Link className={s.referenceCard} to={paths.inspectionStandard(mode, groupId, standard.id)}>
      <div className={s.referenceTop}>
        <div className={s.referenceAvatar}>{standard.reference_path ? <Image /> : <Layers3 />}</div>
        <div className={s.referenceTitle}>
          <strong>{standard.name}</strong>
          <span className={ready ? s.statusPillGood : s.statusPillBad}>
            {ready ? <BadgeCheck /> : <AlertTriangle />} {ready ? "Готов к проверке" : "Нужна разметка"}
          </span>
        </div>
      </div>
      <p>{standard.angle || "Эталонный вид изделия без отдельного описания угла."}</p>
      <div className={s.referenceMeta}>
        <span><Image /> {standard.images_count} images</span>
        <span><ShieldCheck /> {standard.annotated_images_count} annotated</span>
        <span><Clock3 /> {new Date(standard.created_at).toLocaleDateString("ru-RU")}</span>
      </div>
      <span className={s.statusPillMuted}>Открыть station <ArrowRight /></span>
    </Link>
  );
}
