import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { CreateStandard } from "@/page-components/standards/components/create-standard";
import { DeleteStandard } from "@/page-components/standards/components/delete-standard";
import { UpdateStandard } from "@/page-components/standards/components/update-standard";
import { UploadImages } from "@/page-components/standards/components/upload-images";
import { formatDate } from "@/utils/formatDate";
import { CheckCircle2, CircleDashed, Image, Images, ListChecks, Search, SlidersHorizontal, Upload } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import p from "../platform-pages.module.scss";

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
    <div className={p.assetPageV26}>
      <section className={p.assetHeaderV26}>
        <div>
          <span className={p.assetEyebrowV26}><Images /> Assets / References</span>
          <h2>Reference library</h2>
          <p>Эталонные виды изделия. Здесь создаются references, загружаются изображения и открывается очередь разметки.</p>
        </div>
        <CreateStandard groupId={group.id} />
      </section>

      <section className={p.referenceSummaryV26}>
        <div><strong>{group.stats.standards_count}</strong><span>references</span></div>
        <div><strong>{group.stats.images_count}</strong><span>images</span></div>
        <div><strong>{readyCount}</strong><span>ready</span></div>
        <div><strong>{needsLabelCount}</strong><span>needs label</span></div>
      </section>

      <section className={p.referenceToolbarV26}>
        <label>
          <Search />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search references..." />
        </label>
        <div className={p.referenceFiltersV26}>
          {FILTERS.map((item) => (
            <button className={filter === item.key ? p.activeFilterV26 : undefined} key={item.key} type="button" onClick={() => setFilter(item.key)}>
              {item.label}
            </button>
          ))}
        </div>
        <span><SlidersHorizontal /> {filtered.length} shown</span>
      </section>

      <QueryState
        isEmpty={!group.standards.length}
        size="page"
        emptyTitle="References not created"
        emptyDescription="Создай первый эталонный вид, загрузи фотографии и начни разметку."
      >
        <div className={p.referenceLibraryGridV26}>
          {filtered.map((reference) => {
            const labeled = percent(reference.annotated_images_count, reference.images_count);
            const ready = reference.images_count > 0 && labeled === 100;

            return (
              <article className={p.referenceCardV26} key={reference.id}>
                <Link className={p.referenceCardMediaV26} to={paths.standardDetail(group.id, reference.id)}>
                  {reference.reference_path ? <img src={`/storage/${reference.reference_path}`} alt="" /> : <Image />}
                  <b>{reference.is_active ? "active" : "draft"}</b>
                </Link>

                <div className={p.referenceCardBodyV26}>
                  <div>
                    <span className={p.assetEyebrowV26}>{reference.angle || "view"}</span>
                    <h3>{reference.name}</h3>
                    <p>{reference.images_count} images · {reference.annotated_images_count} labeled · created {formatDate(reference.created_at)}</p>
                  </div>

                  <div className={p.referenceProgressV26}>
                    <i style={{ width: `${labeled}%` }} />
                    <span>{labeled}% labeled</span>
                  </div>

                  <div className={p.referenceStatusLineV26}>
                    {ready ? <CheckCircle2 /> : <CircleDashed />}
                    <span>{ready ? "Ready for Train/Inspect" : reference.images_count ? "Continue labeling" : "Upload images first"}</span>
                  </div>
                </div>

                <div className={p.referenceCardActionsV26}>
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
          <div className={p.assetEmptyInlineV26}>
            <Search />
            <strong>No references match filters</strong>
            <span>Попробуй изменить поиск или фильтр.</span>
          </div>
        ) : null}
      </QueryState>
    </div>
  );
}
