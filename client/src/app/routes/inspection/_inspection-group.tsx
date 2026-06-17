import { inspectionModePaths, paths, type InspectionModePath } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import type { GroupStandard } from "@/types/contracts";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Image,
  Layers3,
  ListChecks,
  MousePointer2,
  ShieldCheck,
} from "lucide-react";
import { Link, useParams } from "react-router-dom";
import p from "../platform-pages.module.scss";

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
    <div className={p.inspectStationPageV32}>
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
            <section className={p.inspectScopeHeaderV32}>
              <div>
                <span className={p.eyebrowV27}>Inspect / {currentMode}</span>
                <h1>{group.name}</h1>
                <p>Выбери reference, по которому будут переноситься зоны и проверяться детали.</p>
              </div>
              <div className={p.inspectScopeActionsV32}>
                <Link to={paths.inspectionHistoryGroup(group.id)}><ListChecks /> Runs</Link>
                <Link to={paths.assetReferences(group.id)}><Layers3 /> Assets</Link>
              </div>
            </section>

            <section className={p.inspectReadinessGridV32}>
              <ReadinessCard
                ready={group.standards.length > 0}
                title="References"
                value={group.standards.length}
                hint="Нужен хотя бы один reference"
              />
              <ReadinessCard
                ready={group.stats.segment_classes_count > 0}
                title="Classes"
                value={group.stats.segment_classes_count}
                hint="Выбор классов будет справа"
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
                hint={group.active_model ? `v${group.active_model.version ?? "draft"} · ${group.active_model.architecture}` : "Нет активной модели"}
              />
            </section>

            <section className={p.inspectReferenceShellV32}>
              <div className={p.sectionHeaderV27}>
                <div>
                  <span className={p.eyebrowV27}>References</span>
                  <h2>Выбери эталонный вид</h2>
                </div>
                <span>{group.standards.length} references</span>
              </div>

              <QueryState
                isEmpty={group.standards.length === 0}
                size="block"
                emptyTitle="Нет reference"
                emptyDescription="Сначала добавь reference в Assets и разметь обязательные детали."
              >
                <div className={p.inspectReferenceGridV32}>
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
    <div className={p.inspectReadinessCardV32} data-ready={ready}>
      {ready ? <CheckCircle2 /> : <AlertTriangle />}
      <div>
        <b>{value}</b>
        <span>{title}</span>
        <small>{hint}</small>
      </div>
    </div>
  );
}

function ReferenceCard({ standard, groupId, mode }: { standard: GroupStandard; groupId: string; mode: InspectionModePath }) {
  const annotated = standard.annotated_images_count ?? 0;
  const images = standard.images_count ?? 0;
  const progress = images ? Math.round((annotated / images) * 100) : 0;
  const ready = images > 0 && annotated > 0;

  return (
    <Link className={p.inspectReferenceCardV32} to={paths.inspectionStandard(mode, groupId, standard.id)}>
      <div className={p.inspectReferenceThumbV32}>
        {standard.reference_path ? <img src={standard.reference_path} alt="" /> : <Image />}
        <span data-ready={ready}>{ready ? "ready" : "draft"}</span>
      </div>
      <div className={p.inspectReferenceBodyV32}>
        <div className={p.inspectReferenceTitleV32}>
          <strong>{standard.name}</strong>
          <ArrowRight />
        </div>
        <p>{standard.angle || "Reference view"}</p>
        <div className={p.inspectReferenceMetaV32}>
          <span><Image /> {images} images</span>
          <span><ShieldCheck /> {annotated} labeled</span>
          <span><Clock3 /> {progress}%</span>
        </div>
      </div>
      <div className={p.inspectReferenceHintV32}>
        <MousePointer2 /> Открыть station
      </div>
    </Link>
  );
}
