import clsx from "clsx";
import Button from "@/components/ui/button/button";
import type { GroupDetail } from "@/types/contracts";
import { FolderPlus, ListPlus, Save, Search, Tags, Trash2 } from "lucide-react";
import type { CSSProperties } from "react";
import { useEffect, useMemo, useState } from "react";
import { useManageSegmentGroups } from "../../hooks/use-manage-segment-groups";
import s from "./classes-workspace.module.scss";

type Manager = ReturnType<typeof useManageSegmentGroups>;
type CategoryState = Manager["categories"][number];
type ClassState = Manager["ungroupedClasses"][number];

type CategoryFilter = "all" | "ungrouped" | string;

type ClassLocation = {
  categoryKey: string | null;
  classKey: string;
};

type FlatClass = ClassLocation & {
  categoryName: string;
  item: ClassState;
};

function normalize(value: string) {
  return value.trim().toLowerCase();
}

function flattenClasses(categories: CategoryState[], ungroupedClasses: ClassState[]): FlatClass[] {
  return [
    ...categories.flatMap((category) =>
      category.segmentClasses.map((item) => ({
        categoryKey: category.key,
        classKey: item.key,
        categoryName: category.name || "Без названия",
        item,
      }))
    ),
    ...ungroupedClasses.map((item) => ({
      categoryKey: null,
      classKey: item.key,
      categoryName: "Без категории",
      item,
    })),
  ];
}

function selectionEquals(a: ClassLocation | null, b: ClassLocation | null) {
  return a?.categoryKey === b?.categoryKey && a?.classKey === b?.classKey;
}

function getRowStyle(hue: number): CSSProperties {
  return { "--class-hue": String(hue) } as CSSProperties;
}

export function ClassesWorkspace({ group }: { group: GroupDetail }) {
  const manager = useManageSegmentGroups(group);
  const {
    categories,
    ungroupedClasses,
    fieldErrors,
    saving,
    isDirty,
    categoryActions,
    classActions,
    save,
    reset,
  } = manager;

  const [query, setQuery] = useState("");
  const [savedAt, setSavedAt] = useState<Date | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>("all");
  const [selected, setSelected] = useState<ClassLocation | null>(null);
  const normalizedQuery = normalize(query);

  const flatClasses = useMemo(
    () => flattenClasses(categories, ungroupedClasses),
    [categories, ungroupedClasses]
  );

  const visibleClasses = useMemo(() => {
    return flatClasses.filter(({ categoryKey, categoryName, item }) => {
      const matchesFilter =
        categoryFilter === "all"
          ? true
          : categoryFilter === "ungrouped"
            ? categoryKey === null
            : categoryKey === categoryFilter;

      if (!matchesFilter) return false;
      if (!normalizedQuery) return true;

      return (
        item.name.toLowerCase().includes(normalizedQuery) ||
        categoryName.toLowerCase().includes(normalizedQuery)
      );
    });
  }, [flatClasses, categoryFilter, normalizedQuery]);

  useEffect(() => {
    if (!visibleClasses.length) {
      setSelected(null);
      return;
    }

    const selectedStillVisible = visibleClasses.some((item) => selectionEquals(selected, item));
    if (!selectedStillVisible) {
      const first = visibleClasses[0];
      setSelected({ categoryKey: first.categoryKey, classKey: first.classKey });
    }
  }, [visibleClasses, selected]);

  const selectedCategory =
    categoryFilter !== "all" && categoryFilter !== "ungrouped"
      ? categories.find((item) => item.key === categoryFilter) ?? null
      : null;

  const handleAddClass = () => {
    if (selectedCategory) {
      classActions.addToCategory(selectedCategory.key);
      return;
    }

    if (categoryFilter === "ungrouped") {
      classActions.addUngrouped();
      return;
    }

    if (categories[0]) {
      classActions.addToCategory(categories[0].key);
      return;
    }

    classActions.addUngrouped();
  };

  const handleDeleteClass = (location: ClassLocation, name: string) => {
    if (!window.confirm(`Удалить класс «${name || "без названия"}»?`)) return;
    if (location.categoryKey) {
      classActions.removeFromCategory(location.categoryKey, location.classKey);
    } else {
      classActions.removeUngrouped(location.classKey);
    }
  };

  const handleSave = async () => {
    const ok = await save();
    if (ok) setSavedAt(new Date());
  };

  return (
    <div className={s.page}>
      <section className={s.summary}>
        <article className={s.summaryCard}>
          <span>Классы</span>
          <strong>{flatClasses.length}</strong>
          <small>Всего label-ов</small>
        </article>
        <article className={s.summaryCard}>
          <span>Категории</span>
          <strong>{categories.length}</strong>
          <small>Группы для порядка</small>
        </article>
        <article className={s.summaryCard}>
          <span>Полигоны</span>
          <strong>{group.stats.polygons_count}</strong>
          <small>Разметка в проекте</small>
        </article>
      </section>

      <section className={s.workspace}>
        <aside className={s.sidebar}>
          <div className={s.cardHead}>
            <div>
              <h2>Категории</h2>
              <p>Фильтр списка и группы классов.</p>
            </div>
            <Button className={s.iconAction} variant="ghost" size="icon" icon={FolderPlus} onClick={categoryActions.add} title="Создать категорию" aria-label="Создать категорию" />
          </div>

          <div className={s.categoryList}>
            <Button
              className={clsx(s.categoryItem, categoryFilter === "all" && s.categoryItemActive)}
              variant="plain"
              onClick={() => setCategoryFilter("all")}
            >
              <span>Все классы</span>
              <b>{flatClasses.length}</b>
            </Button>

            {categories.map((category) => (
              <div
                key={category.key}
                className={clsx(s.categoryRow, categoryFilter === category.key && s.categoryRowActive)}
              >
                <Button
                  className={s.categoryItem}
                  variant="plain"
                  onClick={() => setCategoryFilter(category.key)}
                >
                  <span>{category.name || "Без названия"}</span>
                  <b>{category.segmentClasses.length}</b>
                </Button>
                <Button
                  className={s.rowDangerButton}
                  variant="danger"
                  size="icon"
                  icon={Trash2}
                  title="Удалить категорию"
                  aria-label="Удалить категорию"
                  onClick={() => {
                    if (!window.confirm(`Удалить категорию «${category.name || "без названия"}»? Классы перейдут в «Без категории».`)) return;
                    categoryActions.remove(category.key);
                    setCategoryFilter("all");
                  }}
                />
              </div>
            ))}

            <Button
              className={clsx(s.categoryItem, categoryFilter === "ungrouped" && s.categoryItemActive)}
              variant="plain"
              onClick={() => setCategoryFilter("ungrouped")}
            >
              <span>Без категории</span>
              <b>{ungroupedClasses.length}</b>
            </Button>
          </div>

          {selectedCategory ? (
            <label className={s.inlineField}>
              <span>Название выбранной категории</span>
              <input
                value={selectedCategory.name}
                placeholder="Название категории"
                onChange={(event) => categoryActions.updateName(selectedCategory.key, event.target.value)}
              />
              {fieldErrors[`category:${selectedCategory.key}`] ? (
                <small>{fieldErrors[`category:${selectedCategory.key}`]}</small>
              ) : null}
            </label>
          ) : null}
        </aside>

        <div className={s.contentCard}>
          <div className={s.contentHead}>
            <div>
              <span className={s.panelLabel}><Tags /> {selectedCategory?.name || (categoryFilter === "ungrouped" ? "Без категории" : "Классы")}</span>
              <h3>{visibleClasses.length} показано из {flatClasses.length}</h3>
            </div>

            <div className={s.headControls}>
              <label className={s.searchField}>
                <Search />
                <input
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Найти класс или категорию"
                />
              </label>

              <Button className={s.primaryAction} icon={ListPlus} onClick={handleAddClass}>
                {selectedCategory ? "Класс в категорию" : "Добавить класс"}
              </Button>
            </div>
          </div>

          {visibleClasses.length ? (
            <div className={s.table}>
              <div className={s.tableHead}>
                <span>Цвет</span>
                <span>Класс</span>
                <span>Категория</span>
                <span>Оттенок</span>
                <span aria-hidden="true" />
              </div>

              <div className={s.rows}>
                {visibleClasses.map(({ item, categoryKey, classKey }) => {
                  const location = { categoryKey, classKey };
                  return (
                    <div
                      key={classKey}
                      className={clsx(
                        s.classRow,
                        selectionEquals(selected, location) && s.classRowSelected,
                        fieldErrors[`class:${classKey}`] && s.classRowError,
                      )}
                      style={getRowStyle(item.hue)}
                      onClick={() => setSelected(location)}
                    >
                      <div className={s.colorCell}>
                        <i />
                      </div>

                      <label className={s.rowField}>
                        <span>Класс</span>
                        <input
                          value={item.name}
                          placeholder="Название класса"
                          onClick={(event) => event.stopPropagation()}
                          onChange={(event) => classActions.updateName(categoryKey, classKey, event.target.value)}
                        />
                        {fieldErrors[`class:${classKey}`] ? <small>{fieldErrors[`class:${classKey}`]}</small> : null}
                      </label>

                      <label className={s.rowField}>
                        <span>Категория</span>
                        <select
                          value={categoryKey ?? ""}
                          onClick={(event) => event.stopPropagation()}
                          onChange={(event) => {
                            const nextCategoryKey = event.target.value || null;
                            classActions.move(categoryKey, classKey, nextCategoryKey);
                            setSelected({ categoryKey: nextCategoryKey, classKey });
                          }}
                        >
                          <option value="">Без категории</option>
                          {categories.map((category) => (
                            <option key={category.key} value={category.key}>
                              {category.name || "Без названия"}
                            </option>
                          ))}
                        </select>
                      </label>

                      <label className={s.hueField}>
                        <span>{item.hue}°</span>
                        <input
                          className={s.slider}
                          type="range"
                          min={0}
                          max={359}
                          value={item.hue}
                          onClick={(event) => event.stopPropagation()}
                          onChange={(event) => classActions.updateHue(categoryKey, classKey, Number(event.target.value))}
                        />
                      </label>

                      <Button
                        className={s.rowDangerButton}
                        variant="danger"
                        size="icon"
                        icon={Trash2}
                        title="Удалить класс"
                        aria-label="Удалить класс"
                        onClick={(event) => {
                          event.stopPropagation();
                          handleDeleteClass(location, item.name);
                        }}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className={s.emptyState}>
              <Search />
              <strong>Ничего не найдено</strong>
              <span>Попробуй изменить фильтр или поисковый запрос.</span>
            </div>
          )}

          <div className={s.footerBar}>
            <div className={s.saveMeta}>
              {saving ? "Сохранение..." : isDirty ? "Есть несохранённые изменения" : savedAt ? `Сохранено ${savedAt.toLocaleTimeString()}` : "Изменений нет"}
            </div>
            <div className={s.footerActions}>
              <Button className={s.secondaryAction} variant="ghost" disabled={!isDirty || saving} onClick={reset}>
                Отменить
              </Button>
              <Button className={s.primaryAction} icon={Save} disabled={!isDirty || saving} onClick={handleSave}>
                {saving ? "Сохранение..." : "Сохранить"}
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
