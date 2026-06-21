import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { CreateGroup } from "@/page-components/groups/components/create-group";
import { DeleteGroup } from "@/page-components/groups/components/delete-group";
import { UpdateGroup } from "@/page-components/groups/components/update-group";
import type { GroupListItem } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import s from "./_project-assets-strict.module.scss";

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

function setupScore(group: GroupListItem) {
  const checks = [
    group.stats.standards_count > 0,
    group.stats.images_count > 0,
    group.stats.segment_classes_count > 0,
    group.stats.polygons_count > 0,
    group.stats.models_count > 0,
  ];

  return checks.filter(Boolean).length;
}

function setupPercent(group: GroupListItem) {
  return Math.round((setupScore(group) / 5) * 100);
}

function canInspect(group: GroupListItem) {
  return setupScore(group) === 5;
}

function setupHint(group: GroupListItem) {
  if (!group.stats.standards_count) return "нет эталонов";
  if (!group.stats.images_count) return "нет фото";
  if (!group.stats.segment_classes_count) return "нет классов";
  if (!group.stats.polygons_count) return "нет зон контроля";
  if (!group.stats.models_count) return "нет модели";
  return "готово к проверке";
}

function setupLabel(group: GroupListItem) {
  return canInspect(group) ? "Готово" : "Настроить";
}

export function Component() {
  const { data } = useGetGroups();
  const groups = data ?? [];
  const [query, setQuery] = useState("");

  const totals = groups.reduce(
    (acc, group) => ({
      references: acc.references + group.stats.standards_count,
      images: acc.images + group.stats.images_count,
      labeled: acc.labeled + group.stats.annotated_images_count,
      polygons: acc.polygons + group.stats.polygons_count,
      classes: acc.classes + group.stats.segment_classes_count,
      models: acc.models + group.stats.models_count,
      runs: acc.runs + group.stats.inspections_count,
    }),
    { references: 0, images: 0, labeled: 0, polygons: 0, classes: 0, models: 0, runs: 0 },
  );

  const filteredGroups = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return groups;
    return groups.filter((group) =>
      [group.name, group.description ?? ""].some((value) => value.toLowerCase().includes(normalized)),
    );
  }, [groups, query]);

  const normalizedQuery = query.trim();
  const inspectableProjects = groups.filter(canInspect).length;
  const setupProjects = groups.length - inspectableProjects;
  const globalLabeling = percent(totals.labeled, totals.images);
  const resultCountLabel = normalizedQuery ? `${filteredGroups.length} из ${groups.length}` : `${groups.length}`;

  return (
    <div className={s.workspacePage}>
      <div className={`${s.page} ${s.groupsPageV126}`}>
        <header className={s.groupsHeaderV126}>
          <div className={s.groupsTitleV126}>
            <span className={s.eyebrow}>Каталог изделий</span>
            <h1>Изделия</h1>
            <p>{resultCountLabel} в списке · {inspectableProjects} готово · {setupProjects} требует настройки</p>
          </div>

          <div className={s.groupsCreateV126}>
            <CreateGroup />
          </div>

          <label className={`${s.searchBox} ${s.groupsSearchV126}`}>
            <Search />
            <input
              aria-label="Поиск изделий"
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Поиск по названию или описанию"
            />
          </label>
        </header>

        <section className={s.groupsSummaryV126} aria-label="Сводка по изделиям">
          <span><strong>{groups.length}</strong><small>изделий</small></span>
          <span><strong>{inspectableProjects}</strong><small>готово</small></span>
          <span><strong>{setupProjects}</strong><small>настроить</small></span>
          <span><strong>{globalLabeling}%</strong><small>размечено</small></span>
        </section>

        <section className={`${s.contentGrid} ${s.contentGridSingleV87}`}>
          <main className={`${s.panel} ${s.groupsPanelV126}`}>
            <div className={s.groupsListHeadV126}>
              <div>
                <h2>Список изделий</h2>
                <span>{normalizedQuery ? `Найдено ${filteredGroups.length}` : "Строка открывает обзор изделия"}</span>
              </div>
              {normalizedQuery ? (
                <Button className={s.groupsClearSearchV126} variant="ghost" size="sm" onClick={() => setQuery("")}>Сбросить поиск</Button>
              ) : null}
            </div>

            <div className={`${s.panelBody} ${s.groupsPanelBodyV126}`}>
              <QueryState
                isEmpty={!groups.length}
                size="block"
                emptyTitle="Нет изделий"
                emptyDescription="Создай изделие и добавь эталоны, классы и модели."
                action={<CreateGroup />}
              >
                {filteredGroups.length > 0 ? (
                  <div className={s.projectListV126}>
                    {filteredGroups.map((group) => {
                      const score = setupScore(group);
                      const scorePercent = setupPercent(group);
                      const labeled = percent(group.stats.annotated_images_count, group.stats.images_count);
                      const inspectable = canInspect(group);

                      return (
                        <article className={s.projectRowV126} key={group.id}>
                          <Link className={s.projectMainLinkV126} to={paths.groupDetail(group.id)} aria-label={`Открыть обзор изделия ${group.name}`}>
                            <div className={s.projectIdentityV126}>
                              <strong>{group.name}</strong>
                              <p>{group.description || "Без описания"}</p>
                              <small>Создано {formatDate(group.created_at)}</small>
                            </div>

                            <div className={s.projectSetupV126}>
                              <div>
                                <strong className={inspectable ? s.projectSetupReadyV126 : s.projectSetupRequiredV126}>{setupLabel(group)}</strong>
                                <span>{score}/5 · {setupHint(group)}</span>
                              </div>
                              <div
                                className={`${s.progressTrack} ${s.projectProgressV126}`}
                                role="progressbar"
                                aria-label={`Настройка изделия ${group.name}`}
                                aria-valuemin={0}
                                aria-valuemax={5}
                                aria-valuenow={score}
                              >
                                <i style={{ width: `${scorePercent}%` }} />
                              </div>
                            </div>

                            <div className={s.projectStatsV126} aria-label="Состав изделия">
                              <span><strong>{group.stats.standards_count}</strong><small>эталоны</small></span>
                              <span><strong>{group.stats.images_count}</strong><small>фото</small></span>
                              <span><strong>{labeled}%</strong><small>разметка</small></span>
                              <span><strong>{group.stats.segment_classes_count}</strong><small>классы</small></span>
                              <span><strong>{group.stats.models_count}</strong><small>модели</small></span>
                              <span><strong>{group.stats.inspections_count}</strong><small>проверки</small></span>
                            </div>
                          </Link>

                          <div className={s.projectActionsV126}>
                            <UpdateGroup group={group} triggerClassName={`${s.projectCardAction} ${s.projectUtilityActionV126}`} />
                            <DeleteGroup group={group} triggerClassName={`${s.projectCardAction} ${s.projectDeleteActionV126}`} />
                          </div>
                        </article>
                      );
                    })}
                  </div>
                ) : (
                  <div className={`${s.emptyInline} ${s.emptySearchStateV126}`}>
                    <Search />
                    <strong>Ничего не найдено</strong>
                    <span>По запросу «{normalizedQuery}» нет изделий.</span>
                    <Button className={s.groupsClearSearchV126} variant="ghost" size="sm" onClick={() => setQuery("")}>Сбросить поиск</Button>
                  </div>
                )}
              </QueryState>
            </div>
          </main>
        </section>
      </div>
    </div>
  );
}
