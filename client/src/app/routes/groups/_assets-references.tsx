import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { CreateStandard } from "@/page-components/standards/components/create-standard";
import { DeleteStandard } from "@/page-components/standards/components/delete-standard";
import { UpdateStandard } from "@/page-components/standards/components/update-standard";
import { UploadImages } from "@/page-components/standards/components/upload-images";
import { formatDate } from "@/utils/formatDate";
import { CheckCircle2, CircleDashed, Image, Images, ListChecks, Search, SlidersHorizontal } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import s from "./_project-assets-strict.module.scss";

type FilterKey = "all" | "active" | "draft" | "empty" | "labeled" | "needs-label";

const FILTERS: { key: FilterKey; label: string }[] = [
  { key: "all", label: "All" },
  { key: "active", label: "Active" },
  { key: "draft", label: "Draft" },
  { key: "empty", label: "Empty" },
  { key: "labeled", label: "Labeled" },
  { key: "needs-label", label: "Needs label" },
];

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

export function Component() {
  const { group } = useGroupDetailOutletContext();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterKey>("all");

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();

    return group.standards.filter((reference) => {
      const labeled = percent(reference.annotated_images_count, reference.images_count);
      const matchesFilter =
        filter === "all" ||
        (filter === "active" && reference.is_active) ||
        (filter === "draft" && !reference.is_active) ||
        (filter === "empty" && reference.images_count === 0) ||
        (filter === "labeled" && reference.images_count > 0 && labeled === 100) ||
        (filter === "needs-label" && reference.images_count > 0 && labeled < 100);

      const matchesQuery =
        !normalized ||
        [reference.name, reference.angle ?? "", reference.is_active ? "active" : "draft"].some((item) => item.toLowerCase().includes(normalized));

      return matchesFilter && matchesQuery;
    });
  }, [filter, group.standards, query]);

  const readyCount = group.standards.filter((reference) => reference.images_count > 0 && reference.annotated_images_count === reference.images_count).length;
  const needsLabelCount = group.standards.filter((reference) => reference.images_count > reference.annotated_images_count).length;

  return (
    <div className={s.page}>
      <header className={`${s.header} ${s.headerCompact}`}>
        <div>
          <span className={s.eyebrow}><Images /> Assets / References</span>
          <h2>Reference library</h2>
          <p>Эталонные виды изделия. Здесь создаются references, загружаются изображения и открывается очередь разметки.</p>
        </div>
        <div className={s.headerActions}><CreateStandard groupId={group.id} /></div>
      </header>

      <section className={s.summaryGrid}>
        <Metric value={group.stats.standards_count} label="References" hint="total" />
        <Metric value={group.stats.images_count} label="Images" hint="uploaded" />
        <Metric value={readyCount} label="Ready" hint="100% labeled" />
        <Metric value={needsLabelCount} label="Needs label" hint="annotation queue" />
      </section>

      <section className={s.panel}>
        <div className={s.toolbar}>
          <label className={s.searchBox}>
            <Search />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search references..." />
          </label>
          <div className={s.filterBar}>
            {FILTERS.map((item) => (
              <button className={filter === item.key ? s.isActive : undefined} key={item.key} type="button" onClick={() => setFilter(item.key)}>{item.label}</button>
            ))}
          </div>
          <span className={s.statusPill}><SlidersHorizontal /> {filtered.length} shown</span>
        </div>

        <div className={s.panelBody}>
          <QueryState isEmpty={!group.standards.length} size="page" emptyTitle="References not created" emptyDescription="Создай первый эталонный вид, загрузи фотографии и начни разметку.">
            <div className={s.referenceGrid}>
              {filtered.map((reference) => {
                const labeled = percent(reference.annotated_images_count, reference.images_count);
                const ready = reference.images_count > 0 && labeled === 100;

                return (
                  <article className={s.referenceCard} key={reference.id}>
                    <Link className={s.mediaLink} to={paths.standardDetail(group.id, reference.id)}>
                      {reference.reference_path ? <img src={`/storage/${reference.reference_path}`} alt="" /> : <Image />}
                      <b className={reference.is_active ? s.activePill : s.draftPill}>{reference.is_active ? "active" : "draft"}</b>
                    </Link>

                    <div className={s.referenceBody}>
                      <div className={s.cardTitle}>
                        <span className={s.eyebrow}>{reference.angle || "view"}</span>
                        <strong>{reference.name}</strong>
                        <p>{reference.images_count} images · {reference.annotated_images_count} labeled · created {formatDate(reference.created_at)}</p>
                      </div>

                      <div className={s.progressTrack}><i style={{ width: `${labeled}%` }} /></div>

                      <div className={ready ? s.donePill : s.openPill}>
                        {ready ? <CheckCircle2 /> : <CircleDashed />}
                        <span>{ready ? "Ready" : reference.images_count ? "Needs label" : "Upload first"}</span>
                      </div>
                    </div>

                    <div className={s.referenceActions}>
                      <Link to={paths.standardDetail(group.id, reference.id)}>Open</Link>
                      <UploadImages groupId={group.id} standardId={reference.id} />
                      <Link to={paths.inspectionStandard("photo", group.id, reference.id)}><ListChecks /> Inspect</Link>
                      <UpdateStandard standard={reference} />
                      <DeleteStandard groupId={group.id} id={reference.id} name={reference.name} />
                    </div>
                  </article>
                );
              })}
            </div>

            {group.standards.length > 0 && filtered.length === 0 ? (
              <div className={s.emptyInline}><Search /><strong>No references match filters</strong><span>Попробуй изменить поиск или фильтр.</span></div>
            ) : null}
          </QueryState>
        </div>
      </section>
    </div>
  );
}

function Metric({ value, label, hint }: { value: number; label: string; hint: string }) {
  return (
    <div className={s.metricCard}>
      <Images />
      <div><span>{label}</span><strong>{value}</strong><small>{hint}</small></div>
    </div>
  );
}
