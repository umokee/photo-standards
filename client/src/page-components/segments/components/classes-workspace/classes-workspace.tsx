import Button from "@/components/ui/button/button";
import type { GroupDetail } from "@/types/contracts";
import clsx from "clsx";
import {
  ChevronDown,
  ChevronRight,
  Circle,
  FolderPlus,
  Info,
  ListPlus,
  Palette,
  Save,
  Search,
  Tags,
  Trash2,
} from "lucide-react";
import type { CSSProperties } from "react";
import { useEffect, useMemo, useState } from "react";
import { useManageSegmentGroups } from "../../hooks/use-manage-segment-groups";
import s from "./classes-workspace.module.scss";

type Manager = ReturnType<typeof useManageSegmentGroups>;
type CategoryState = Manager["categories"][number];
type ClassState = Manager["ungroupedClasses"][number];

type Selection = {
  categoryKey: string | null;
  classKey: string;
};

type FlatClass = Selection & {
  item: ClassState;
  categoryName: string;
};

function normalize(value: string) {
  return value.trim().toLowerCase();
}

function classColor(hue: number) {
  return `hsl(${hue}, 70%, 52%)`;
}

function flattenClasses(categories: CategoryState[], ungroupedClasses: ClassState[]): FlatClass[] {
  return [
    ...categories.flatMap((category) =>
      category.segmentClasses.map((item) => ({
        item,
        classKey: item.key,
        categoryKey: category.key,
        categoryName: category.name || "Без названия",
      }))
    ),
    ...ungroupedClasses.map((item) => ({
      item,
      classKey: item.key,
      categoryKey: null,
      categoryName: "Без категории",
    })),
  ];
}

function isSameSelection(a: Selection | null, b: Selection | null) {
  return a?.classKey === b?.classKey && a?.categoryKey === b?.categoryKey;
}

export function ClassesWorkspace({ group }: { group: GroupDetail }) {
  const manager = useManageSegmentGroups(group);
  const {
    categories,
    ungroupedClasses,
    fieldErrors,
    saving,
    categoryActions,
    classActions,
    save,
  } = manager;

  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Selection | null>(null);
  const [savedAt, setSavedAt] = useState<Date | null>(null);
  const normalizedQuery = normalize(query);

  const flatClasses = useMemo(
    () => flattenClasses(categories, ungroupedClasses),
    [categories, ungroupedClasses]
  );

  const active = flatClasses.find(
    (item) => item.classKey === selected?.classKey && item.categoryKey === selected.categoryKey
  ) ?? flatClasses[0] ?? null;

  useEffect(() => {
    if (!flatClasses.length) {
      setSelected(null);
      return;
    }

    const selectedStillExists = flatClasses.some(
      (item) => item.classKey === selected?.classKey && item.categoryKey === selected.categoryKey
    );

    if (!selectedStillExists) {
      setSelected({ categoryKey: flatClasses[0].categoryKey, classKey: flatClasses[0].classKey });
    }
  }, [flatClasses, selected]);

  const visibleCategories = useMemo(
    () =>
      categories
        .map((category) => ({
          ...category,
          segmentClasses: category.segmentClasses.filter(
            (item) =>
              !normalizedQuery ||
              item.name.toLowerCase().includes(normalizedQuery) ||
              category.name.toLowerCase().includes(normalizedQuery)
          ),
        }))
        .filter((category) => category.segmentClasses.length || !normalizedQuery),
    [categories, normalizedQuery]
  );

  const visibleUngrouped = useMemo(
    () =>
      ungroupedClasses.filter((item) => !normalizedQuery || item.name.toLowerCase().includes(normalizedQuery)),
    [ungroupedClasses, normalizedQuery]
  );

  const visibleCount =
    visibleCategories.reduce((sum, category) => sum + category.segmentClasses.length, 0) +
    visibleUngrouped.length;

  const handleAddClass = () => {
    if (active?.categoryKey) {
      classActions.addToCategory(active.categoryKey);
      return;
    }

    if (categories[0]) {
      classActions.addToCategory(categories[0].key);
      return;
    }

    classActions.addUngrouped();
  };

  const handleSave = async () => {
    const ok = await save();
    if (ok) {
      setSavedAt(new Date());
    }
  };

  const empty = !flatClasses.length;

  return (
    <div className={s.page}>
      <section className={s.header}>
        <div>
          <span className={s.eyebrow}><Tags /> Assets / Classes</span>
          <h2>Классы деталей</h2>
          <p>
            Структура обязательных компонентов изделия. Здесь создаются категории, классы и цвета,
            которые затем используются в редакторе, обучении и проверке.
          </p>
        </div>

        <div className={s.headerStats}>
          <span><b>{flatClasses.length}</b> классов</span>
          <span><b>{categories.length}</b> категорий</span>
          <span><b>{group.stats.polygons_count}</b> полигонов</span>
        </div>
      </section>

      <section className={s.workspace}>
        <div className={s.listPane}>
          <div className={s.toolbar}>
            <label className={s.search}>
              <Search />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Поиск классов..." />
            </label>

            <button type="button" className={s.toolButton} onClick={categoryActions.add}>
              <FolderPlus /> Категория
            </button>
            <button type="button" className={s.toolButtonPrimary} onClick={handleAddClass}>
              <ListPlus /> Класс
            </button>
          </div>

          <div className={s.listMeta}>
            <span>{visibleCount} показано · {flatClasses.length} всего</span>
            {savedAt ? <span>Сохранено {savedAt.toLocaleTimeString()}</span> : <span>Изменения сохраняются вручную</span>}
          </div>

          {empty ? (
            <div className={s.emptyState}>
              <Tags />
              <strong>Классы ещё не настроены</strong>
              <span>Создай категории и классы деталей: вентиль, манометр, рычаг, табличка и другие элементы контроля.</span>
              <button type="button" onClick={handleAddClass}>Создать первый класс</button>
            </div>
          ) : null}

          {!empty && visibleCount === 0 ? (
            <div className={s.emptyState}>
              <Search />
              <strong>Ничего не найдено</strong>
              <span>Попробуй изменить поисковый запрос.</span>
            </div>
          ) : null}

          <div className={s.categoryList}>
            {visibleCategories.map((category) => (
              <CategoryBlock
                key={category.key}
                category={category}
                selected={selected}
                fieldErrors={fieldErrors}
                onSelect={setSelected}
                onAddClass={() => classActions.addToCategory(category.key)}
                onToggle={() => categoryActions.toggle(category.key)}
                onRemove={() => {
                  if (window.confirm(`Удалить категорию «${category.name || "без названия"}»? Классы перейдут в блок «Без категории».`)) {
                    categoryActions.remove(category.key);
                  }
                }}
                onRename={(value) => categoryActions.updateName(category.key, value)}
              />
            ))}

            {(visibleUngrouped.length > 0 || !normalizedQuery) && (
              <UngroupedBlock
                items={visibleUngrouped}
                selected={selected}
                fieldErrors={fieldErrors}
                onSelect={setSelected}
                onAddClass={classActions.addUngrouped}
              />
            )}
          </div>
        </div>

        <aside className={s.inspectorPane}>
          <div className={s.inspectorHeader}>
            <span><Palette /> Параметры класса</span>
            {active ? <b style={{ "--class-hue": active.item.hue } as CSSProperties} /> : null}
          </div>

          {active ? (
            <div className={s.inspectorBody}>
              <label className={s.field}>
                <span>Название</span>
                <input
                  value={active.item.name}
                  placeholder="Например: Винт M16"
                  onChange={(event) => classActions.updateName(active.categoryKey, active.classKey, event.target.value)}
                />
                {fieldErrors[`class:${active.classKey}`] ? <small>{fieldErrors[`class:${active.classKey}`]}</small> : null}
              </label>

              <label className={s.field}>
                <span>Категория</span>
                <select
                  value={active.categoryKey ?? ""}
                  onChange={(event) => {
                    const nextCategoryKey = event.target.value || null;
                    classActions.move(active.categoryKey, active.classKey, nextCategoryKey);
                    setSelected({ categoryKey: nextCategoryKey, classKey: active.classKey });
                  }}
                >
                  <option value="">Без категории</option>
                  {categories.map((category) => (
                    <option key={category.key} value={category.key}>{category.name || "Без названия"}</option>
                  ))}
                </select>
              </label>

              <label className={s.field}>
                <span>Цвет · hue {active.item.hue}</span>
                <div className={s.colorBox} style={{ "--class-hue": active.item.hue } as CSSProperties}>
                  <i />
                  <input
                    type="range"
                    min={0}
                    max={359}
                    value={active.item.hue}
                    onChange={(event) => classActions.updateHue(active.categoryKey, active.classKey, Number(event.target.value))}
                  />
                </div>
              </label>

              <div className={s.usageBox}>
                <Info />
                <div>
                  <strong>Где используется</strong>
                  <span>
                    Класс появится в редакторе разметки, попадёт в Train как label и будет доступен в Inspect как проверяемая деталь.
                  </span>
                </div>
              </div>

              <button
                type="button"
                className={s.deleteClassButton}
                onClick={() => {
                  if (!window.confirm(`Удалить класс «${active.item.name || "без названия"}»?`)) return;
                  if (active.categoryKey) {
                    classActions.removeFromCategory(active.categoryKey, active.classKey);
                  } else {
                    classActions.removeUngrouped(active.classKey);
                  }
                }}
              >
                <Trash2 /> Удалить класс
              </button>
            </div>
          ) : (
            <div className={s.noSelection}>
              <Circle />
              <strong>Выбери класс</strong>
              <span>После выбора справа появятся название, категория и цвет.</span>
            </div>
          )}

          <div className={s.saveBar}>
            <Button variant="ghost" size="sm" onClick={() => window.location.reload()}>
              Отменить
            </Button>
            <Button icon={Save} disabled={saving} onClick={handleSave}>
              {saving ? "Сохранение..." : "Сохранить"}
            </Button>
          </div>
        </aside>
      </section>
    </div>
  );
}

function CategoryBlock({
  category,
  selected,
  fieldErrors,
  onSelect,
  onAddClass,
  onToggle,
  onRemove,
  onRename,
}: {
  category: CategoryState;
  selected: Selection | null;
  fieldErrors: Record<string, string>;
  onSelect: (selection: Selection) => void;
  onAddClass: () => void;
  onToggle: () => void;
  onRemove: () => void;
  onRename: (value: string) => void;
}) {
  const firstHue = category.segmentClasses[0]?.hue ?? 210;

  return (
    <section className={s.categoryCard} style={{ "--class-hue": firstHue } as CSSProperties}>
      <header className={s.categoryHeader}>
        <button type="button" className={s.collapseButton} onClick={onToggle}>
          {category.collapsed ? <ChevronRight /> : <ChevronDown />}
        </button>

        <input
          value={category.name}
          placeholder="Название категории"
          onChange={(event) => onRename(event.target.value)}
        />

        <span>{category.segmentClasses.length}</span>

        <button type="button" className={s.iconButton} onClick={onAddClass} title="Добавить класс">
          <ListPlus />
        </button>
        <button type="button" className={s.iconButtonDanger} onClick={onRemove} title="Удалить категорию">
          <Trash2 />
        </button>
      </header>

      {fieldErrors[`category:${category.key}`] ? <small className={s.inlineError}>{fieldErrors[`category:${category.key}`]}</small> : null}

      {!category.collapsed ? (
        <div className={s.classRows}>
          {category.segmentClasses.map((item) => (
            <ClassRow
              key={item.key}
              item={item}
              selected={isSameSelection(selected, { categoryKey: category.key, classKey: item.key })}
              error={fieldErrors[`class:${item.key}`]}
              onSelect={() => onSelect({ categoryKey: category.key, classKey: item.key })}
            />
          ))}

          {!category.segmentClasses.length ? <div className={s.categoryEmpty}>В категории пока нет классов.</div> : null}
        </div>
      ) : null}
    </section>
  );
}

function UngroupedBlock({
  items,
  selected,
  fieldErrors,
  onSelect,
  onAddClass,
}: {
  items: ClassState[];
  selected: Selection | null;
  fieldErrors: Record<string, string>;
  onSelect: (selection: Selection) => void;
  onAddClass: () => void;
}) {
  return (
    <section className={s.categoryCard} style={{ "--class-hue": 210 } as CSSProperties}>
      <header className={s.categoryHeader}>
        <span className={s.collapseButton}><Circle /></span>
        <input value="Без категории" readOnly />
        <span>{items.length}</span>
        <button type="button" className={s.iconButton} onClick={onAddClass} title="Добавить класс без категории">
          <ListPlus />
        </button>
      </header>

      <div className={s.classRows}>
        {items.map((item) => (
          <ClassRow
            key={item.key}
            item={item}
            selected={isSameSelection(selected, { categoryKey: null, classKey: item.key })}
            error={fieldErrors[`class:${item.key}`]}
            onSelect={() => onSelect({ categoryKey: null, classKey: item.key })}
          />
        ))}

        {!items.length ? <div className={s.categoryEmpty}>Классов без категории нет.</div> : null}
      </div>
    </section>
  );
}

function ClassRow({ item, selected, error, onSelect }: { item: ClassState; selected: boolean; error?: string; onSelect: () => void }) {
  return (
    <button
      type="button"
      className={clsx(s.classRow, selected && s.classRowActive, error && s.classRowError)}
      style={{ "--class-hue": item.hue } as CSSProperties}
      onClick={onSelect}
    >
      <i />
      <strong>{item.name || "Новый класс"}</strong>
      <small>{error || `hue ${item.hue}`}</small>
    </button>
  );
}
