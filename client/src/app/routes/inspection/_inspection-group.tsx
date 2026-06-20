import { inspectionModePaths, paths, type InspectionModePath } from "@/app/paths";
import { RouteHero } from "@/components/layouts/route-hero/route-hero";
import { RoutePanel } from "@/components/layouts/route-panel/route-panel";
import { ReadinessCard as SharedReadinessCard } from "@/components/ui/readiness-card/readiness-card";
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
            <RouteHero
              eyebrow="Inspect"
              icon={ShieldCheck}
              title={group.name}
              description="Выбери эталонный вид для проверки."
              actionsClassName={s.scopeActions}
              actions={(
                <>
                <Link to={paths.inspectionHistoryGroup(group.id)}><ListChecks /> История</Link>
                <Link to={paths.assetReferences(group.id)}><Layers3 /> Assets</Link>
                </>
              )}
            />

            <section className={s.readinessGrid}>
              <ReadinessCard
                ready={group.standards.length > 0}
                title="Эталоны"
                value={group.standards.length}
                hint="минимум один вид"
              />
              <ReadinessCard
                ready={group.stats.segment_classes_count > 0}
                title="Классы"
                value={group.stats.segment_classes_count}
                hint="детали для проверки"
              />
              <ReadinessCard
                ready={group.stats.polygons_count > 0}
                title="Полигоны"
                value={group.stats.polygons_count}
                hint="ожидаемые зоны"
              />
              <ReadinessCard
                ready={Boolean(group.active_model)}
                title="Модель"
                value={group.active_model ? 1 : 0}
                hint={
                  group.active_model
                    ? `v${group.active_model.version ?? "draft"} · ${group.active_model.architecture}`
                    : "Нет активной модели"
                }
              />
            </section>

            <RoutePanel
              className={s.referenceShell}
              headerClassName={s.sectionHeader}
              headingClassName={s.sectionHeading}
              kickerClassName={s.eyebrow}
              titleClassName={s.sectionTitle}
              actionsClassName={s.sectionCount}
              kicker={<><Image /> Эталоны</>}
              title="Выбери эталон"
              actions={<>{group.standards.length}</>}
            >
              <QueryState
                isEmpty={group.standards.length === 0}
                size="block"
                emptyTitle="Нет эталонов"
                emptyDescription="Добавь эталон в Assets и разметь детали."
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
            </RoutePanel>
          </>
        ) : null}
      </QueryState>
    </div>
  );
}

function ReadinessCard({ ready, title, value, hint }: { ready: boolean; title: string; value: number; hint: string }) {
  return (
    <SharedReadinessCard
      ready={ready}
      title={title}
      value={value}
      hint={hint}
      className={s.readinessCard}
      topClassName={s.readinessTop}
      iconWrapClassName={s.readinessIcon}
    />
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
        <span><Image /> {standard.images_count} фото</span>
        <span><ShieldCheck /> {standard.annotated_images_count} размечено</span>
        <span><Clock3 /> {new Date(standard.created_at).toLocaleDateString("ru-RU")}</span>
      </div>
      <span className={s.statusPillMuted}>Открыть <ArrowRight /></span>
    </Link>
  );
}
