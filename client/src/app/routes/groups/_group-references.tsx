import { paths } from "@/app/paths";
import ImageWithFallback from "@/components/ui/image-with-fallback/image-with-fallback";
import QueryState from "@/components/ui/query-state/query-state";
import { CreateStandard } from "@/page-components/standards/components/create-standard";
import { DeleteStandard } from "@/page-components/standards/components/delete-standard";
import { UpdateStandard } from "@/page-components/standards/components/update-standard";
import { UploadImages } from "@/page-components/standards/components/upload-images";
import type { GroupStandard } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import clsx from "clsx";
import { ArrowRight, CheckCircle2, CircleDashed, Filter, Image, Images, ListChecks, Search, ShieldCheck, Tags, Upload } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import p from "../platform-pages.module.scss";

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
  if (filter === "all") return "All";
  if (filter === "active") return "Active";
  if (filter === "draft") return "Draft";
  if (filter === "empty") return "Empty";
  if (filter === "labeled") return "Labeled";
  return "Needs label";
}

function getReferenceStatus(reference: GroupStandard) {
  if (!reference.images_count) return { label: "empty", tone: "warning" as const };
  if (reference.annotated_images_count >= reference.images_count) return { label: "ready", tone: "success" as const };
  return { label: "labeling", tone: "info" as const };
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
    <div className={p.referencesPageV25}>
      <section className={p.assetsCommandBarV25}>
        <div>
          <span className={p.eyebrow}>Assets / References</span>
          <h1>References</h1>
          <p>Эталонные виды изделия. Каждый reference хранит фотографии, разметку и точку входа в редактор.</p>
        </div>
        <CreateStandard groupId={group.id} />
      </section>

      <section className={p.referenceStatsV25}>
        <Stat icon={Images} value={group.standards.length} label="References" hint={`${readyCount} ready · ${emptyCount} empty`} />
        <Stat icon={ListChecks} value={needsLabelCount} label="Need labeling" hint="images without polygons" />
        <Stat icon={Tags} value={group.stats.segment_classes_count} label="Classes" hint={`${group.stats.segment_class_groups_count} categories`} />
        <Stat icon={ShieldCheck} value={labeledPercent} label="Labeled" hint={`${group.stats.annotated_images_count}/${group.stats.images_count} images`} suffix="%" />
      </section>

      <section className={p.referenceToolbarV25}>
        <label>
          <Search />
          <input value={query} placeholder="Search references..." onChange={(event) => setQuery(event.target.value)} />
        </label>
        <div className={p.filterPillsV25}>
          <Filter />
          {filters.map((item) => (
            <button key={item} type="button" className={clsx(item === filter && p.active)} onClick={() => setFilter(item)}>
              {filterLabel(item)}
            </button>
          ))}
        </div>
      </section>

      <QueryState
        isEmpty={!references.length}
        emptyTitle={group.standards.length ? "No matching references" : "No references yet"}
        emptyDescription={group.standards.length ? "Измени поиск или фильтр." : "Создай первый эталонный вид изделия."}
      >
        <section className={p.referenceGridV25}>
          {references.map((reference) => (
            <ReferenceCard key={reference.id} reference={reference} groupId={group.id} />
          ))}
        </section>
      </QueryState>
    </div>
  );
}

function Stat({ icon: Icon, value, label, hint, suffix = "" }: { icon: typeof Images; value: number; label: string; hint: string; suffix?: string }) {
  return (
    <div className={p.referenceStatCardV25}>
      <Icon />
      <div>
        <strong>{value}{suffix}</strong>
        <span>{label}</span>
        <small>{hint}</small>
      </div>
    </div>
  );
}

function ReferenceCard({ reference, groupId }: { reference: GroupStandard; groupId: string }) {
  const status = getReferenceStatus(reference);
  const labeled = percent(reference.annotated_images_count, reference.images_count);

  return (
    <article className={p.referenceCardV25}>
      <Link className={p.referencePreviewV25} to={paths.standardDetail(groupId, reference.id)}>
        <ImageWithFallback src={reference.reference_path ? `/storage/${reference.reference_path}` : null} iconSize={28} />
        <span className={clsx(p.referenceStatusV25, p[status.tone])}>{status.label}</span>
        {reference.is_active ? <b>active</b> : null}
      </Link>

      <div className={p.referenceCardBodyV25}>
        <div>
          <span className={p.eyebrow}>Reference / {reference.angle || "view"}</span>
          <h3>{reference.name}</h3>
          <p>{reference.images_count} images · {reference.annotated_images_count} labeled · created {formatDate(reference.created_at)}</p>
        </div>

        <div className={p.referenceProgressV25}>
          <span><CheckCircle2 /> {labeled}% labeled</span>
          <i><b style={{ width: `${labeled}%` }} /></i>
        </div>

        <div className={p.referenceTodoV25}>
          {!reference.images_count ? <span><Upload /> Upload images</span> : null}
          {reference.images_count > 0 && reference.annotated_images_count < reference.images_count ? <span><CircleDashed /> Finish annotation</span> : null}
          {reference.images_count > 0 && reference.annotated_images_count >= reference.images_count ? <span><CheckCircle2 /> Ready for inspect</span> : null}
        </div>

        <div className={p.referenceCardActionsV25}>
          <Link to={paths.standardDetail(groupId, reference.id)}>Open <ArrowRight /></Link>
          <UploadImages groupId={groupId} standardId={reference.id} />
          <Link to={paths.inspectionStandard("photo", groupId, reference.id)}>Use in Inspect</Link>
          <UpdateStandard standard={reference} />
          <DeleteStandard groupId={groupId} id={reference.id} name={reference.name} />
        </div>
      </div>
    </article>
  );
}
