import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import ImageWithFallback from "@/components/ui/image-with-fallback/image-with-fallback";
import QueryState from "@/components/ui/query-state/query-state";
import { CreateStandard } from "@/page-components/standards/components/create-standard";
import { DeleteStandard } from "@/page-components/standards/components/delete-standard";
import { UpdateStandard } from "@/page-components/standards/components/update-standard";
import { UploadImages } from "@/page-components/standards/components/upload-images";
import type { GroupStandard } from "@/types/contracts";
import clsx from "clsx";
import { ArrowRight, CheckCircle2, CircleDashed, Filter, Images, ListChecks, Search, ShieldCheck, Tags, Upload } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import p from "./_group-references.module.scss";

type ReferenceFilter = "all" | "active" | "draft" | "empty" | "labeled" | "needs-label";

const filters: ReferenceFilter[] = ["all", "active", "draft", "empty", "labeled", "needs-label"];

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

function matchesFilter(reference: GroupStandard, filter: ReferenceFilter) {
  if (filter === "all") return true;
  if (filter === "active") return reference.is_active;
  if (filter === "draft") return !reference.is_active;
  if (filter === "empty") return reference.images_count === 0;
  if (filter === "labeled") return reference.images_count > 0 && reference.annotated_images_count === reference.images_count;
  if (filter === "needs-label") return reference.images_count > 0 && reference.annotated_images_count < reference.images_count;
  return true;
}

function filterLabel(filter: ReferenceFilter) {
  if (filter === "all") return "Все";
  if (filter === "active") return "Активные";
  if (filter === "draft") return "Черновики";
  if (filter === "empty") return "Без фото";
  if (filter === "labeled") return "Размечены";
  return "Нужна разметка";
}

function getReferenceStatus(reference: GroupStandard) {
  if (!reference.images_count) return { label: "без фото", tone: "warning" as const };
  if (reference.annotated_images_count >= reference.images_count) return { label: "готово", tone: "success" as const };
  return { label: "разметка", tone: "info" as const };
}

export function Component() {
  const { group } = useGroupDetailOutletContext();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<ReferenceFilter>("all");

  const normalizedQuery = query.trim().toLowerCase();
  const references = useMemo(() => {
    return group.standards
      .filter((reference) => matchesFilter(reference, filter))
      .filter((reference) => {
        if (!normalizedQuery) return true;
        return `${reference.name} ${reference.angle ?? ""}`.toLowerCase().includes(normalizedQuery);
      })
      .sort((a, b) => Number(b.is_active) - Number(a.is_active) || b.images_count - a.images_count || a.name.localeCompare(b.name));
  }, [filter, group.standards, normalizedQuery]);

  const readyCount = group.standards.filter((reference) => reference.images_count > 0 && reference.annotated_images_count >= reference.images_count).length;
  const needsLabelCount = group.standards.filter((reference) => reference.images_count > 0 && reference.annotated_images_count < reference.images_count).length;
  const emptyCount = group.standards.filter((reference) => reference.images_count === 0).length;
  const labeledPercent = percent(group.stats.annotated_images_count, group.stats.images_count);

  return (
    <div className={p.referencesPageV128}>
      <header className={p.referenceHeaderV128}>
        <div>
          <span className={p.eyebrowV128}><Images /> Эталонные виды</span>
          <h1>Эталоны</h1>
          <p>Фото, ракурсы и разметка для проверки изделия.</p>
        </div>
        <CreateStandard groupId={group.id} />
      </header>

      <section className={p.referenceSummaryV128} aria-label="Сводка по эталонам">
        <SummaryItem icon={Images} value={group.standards.length} label="эталонов" hint={`${readyCount} готово · ${emptyCount} без фото`} />
        <SummaryItem icon={ListChecks} value={needsLabelCount} label="разметить" hint="фото без полигонов" />
        <SummaryItem icon={Tags} value={group.stats.segment_classes_count} label="классов" hint={`${group.stats.segment_class_groups_count} групп`} />
        <SummaryItem icon={ShieldCheck} value={`${labeledPercent}%`} label="готово" hint={`${group.stats.annotated_images_count}/${group.stats.images_count} фото`} />
      </section>

      <section className={p.referenceToolbarV128}>
        <label className={p.referenceSearchV128}>
          <Search />
          <span className={p.searchLabelV128}>Поиск эталонов</span>
          <input value={query} placeholder="Поиск по названию или виду..." aria-label="Поиск эталонов" onChange={(event) => setQuery(event.target.value)} />
        </label>

        <div className={p.filterPillsV128}>
          <Filter />
          {filters.map((item) => (
            <Button
              key={item}
              variant={item === filter ? "primary" : "ghost"}
              size="sm"
              aria-pressed={item === filter}
              className={clsx(p.referenceFilterButtonV131, item === filter && p.active)}
              onClick={() => setFilter(item)}
            >
              {filterLabel(item)}
            </Button>
          ))}
        </div>
      </section>

      <QueryState
        isEmpty={!references.length}
        emptyTitle={group.standards.length ? "Ничего не найдено" : "Нет эталонов"}
        emptyDescription={group.standards.length ? "Измени поиск или фильтр." : "Создай первый эталонный вид изделия."}
      >
        <section className={p.referenceListV128}>
          {references.map((reference) => (
            <ReferenceRow key={reference.id} reference={reference} groupId={group.id} />
          ))}
        </section>
      </QueryState>
    </div>
  );
}

function SummaryItem({ icon: Icon, value, label, hint }: { icon: typeof Images; value: number | string; label: string; hint: string }) {
  return (
    <article className={p.referenceSummaryItemV128}>
      <Icon />
      <div>
        <strong>{value}</strong>
        <span>{label}</span>
        <small>{hint}</small>
      </div>
    </article>
  );
}

function ReferenceRow({ reference, groupId }: { reference: GroupStandard; groupId: string }) {
  const status = getReferenceStatus(reference);
  const labeled = percent(reference.annotated_images_count, reference.images_count);

  return (
    <article className={p.referenceRowV128}>
      <Link className={p.referencePreviewV128} to={paths.standardDetail(groupId, reference.id)} aria-label={`Открыть эталон ${reference.name}`}>
        <ImageWithFallback src={reference.reference_path ? `/storage/${reference.reference_path}` : null} iconSize={24} />
      </Link>

      <div className={p.referenceInfoV128}>
        <div className={p.referenceTitleRowV128}>
          <div>
            <span className={p.referenceAngleV128}>{reference.angle || "вид изделия"}</span>
            <h3>{reference.name}</h3>
          </div>
          <div className={p.referenceBadgesV128}>
            <span className={clsx(p.referenceStatusV128, p[status.tone])}>{status.label}</span>
            {reference.is_active ? <span className={p.referenceActiveV128}>активен</span> : null}
          </div>
        </div>

        <div className={p.referenceMetaV128}>
          <span>{reference.images_count} фото</span>
          <span>{reference.annotated_images_count} размечено</span>
          <span>{labeled}% готово</span>
        </div>

        <div className={p.referenceProgressV128} role="progressbar" aria-label={`Разметка эталона ${reference.name}`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={labeled}>
          <i><b style={{ width: `${labeled}%` }} /></i>
          <small>{reference.images_count ? `${reference.annotated_images_count}/${reference.images_count}` : "нет фото"}</small>
        </div>
      </div>

      <div className={p.referenceNextV128}>
        {!reference.images_count ? <span><Upload /> Загрузить фото</span> : null}
        {reference.images_count > 0 && reference.annotated_images_count < reference.images_count ? <span><CircleDashed /> Доделать разметку</span> : null}
        {reference.images_count > 0 && reference.annotated_images_count >= reference.images_count ? <span><CheckCircle2 /> Готов к проверке</span> : null}
      </div>

      <div className={p.referenceActionsV128}>
        <Link className={p.referenceOpenActionV128} to={paths.standardDetail(groupId, reference.id)}>Открыть <ArrowRight /></Link>
        <Link className={p.referenceCheckActionV128} to={paths.inspectionStandard("photo", groupId, reference.id)}>Проверить</Link>
        <div className={p.referenceUtilityActionsV128} role="group" aria-label="Дополнительные действия эталона">
          <UploadImages groupId={groupId} standardId={reference.id} triggerClassName={p.referenceActionButtonV128} />
          <UpdateStandard standard={reference} triggerClassName={p.referenceActionButtonV128} />
          <DeleteStandard groupId={groupId} id={reference.id} name={reference.name} triggerClassName={p.referenceDangerButtonV128} />
        </div>
      </div>
    </article>
  );
}
