import Button from "@/components/ui/button/button";
import { ColorPicker } from "@/components/ui/color-picker/color-picker";
import { Modal, useModalClose } from "@/components/ui/modal/modal";
import { GroupDetail } from "@/types/contracts";
import clsx from "clsx";
import { ChevronRight, Plus, Save, X } from "lucide-react";
import type { CSSProperties } from "react";
import { useManageSegmentGroups } from "../../hooks/use-manage-segment-groups";
import s from "./manage-segment-groups.module.scss";

interface Props {
  group: GroupDetail;
  compact?: boolean;
  standardId?: string;
  imageId?: string;
}

interface ClassState {
  key: string;
  name: string;
  hue: number;
}

interface CategoryState {
  key: string;
  name: string;
  collapsed: boolean;
  segmentClasses: ClassState[];
}

interface CategoryActions {
  add: () => void;
  toggle: (key: string) => void;
  updateName: (key: string, value: string) => void;
  remove: (key: string) => void;
}

interface ClassActions {
  addToCategory: (categoryKey: string) => void;
  addUngrouped: () => void;
  removeFromCategory: (categoryKey: string, classKey: string) => void;
  removeUngrouped: (classKey: string) => void;
  updateName: (categoryKey: string | null, classKey: string, value: string) => void;
  updateHue: (categoryKey: string | null, classKey: string, value: number) => void;
}

interface ClassRowProps {
  item: ClassState;
  categoryKey: string | null;
  error?: string;
  isColorOpen: boolean;
  toggleColorPicker: (key: string) => void;
  closeColorPicker: () => void;
  classActions: ClassActions;
}

const ClassRow = ({
  item,
  categoryKey,
  error,
  isColorOpen,
  toggleColorPicker,
  closeColorPicker,
  classActions,
}: ClassRowProps) => {
  const color = `hsl(${item.hue}, 65%, 55%)`;

  const handleRemove = () => {
    if (!window.confirm(`Удалить класс "${item.name || "без названия"}"?`)) {
      return;
    }

    if (isColorOpen) {
      closeColorPicker();
    }

    if (categoryKey) {
      classActions.removeFromCategory(categoryKey, item.key);
      return;
    }

    classActions.removeUngrouped(item.key);
  };

  return (
    <>
      <div className={s.segmentRow}>
        <div className={s.colorField}>
          <button
            type="button"
            className={clsx(s.colorTrigger, isColorOpen && s.colorOpen)}
            aria-label="Изменить цвет класса"
            onClick={(e) => {
              e.preventDefault();
              toggleColorPicker(item.key);
            }}
          >
            <span className={s.colorSwatch} style={{ background: color }} />
          </button>
        </div>

        <input
          className={clsx(s.nameInput, error && s.nameInputError)}
          value={item.name}
          placeholder="Название класса"
          onChange={(e) => classActions.updateName(categoryKey, item.key, e.target.value)}
        />

        <button type="button" className={clsx(s.iconButton, s.removeButton)} onClick={handleRemove}>
          <X size={14} />
        </button>
      </div>

      {error ? <div className={s.inlineError}>{error}</div> : null}

      {isColorOpen && (
        <div className={s.colorPanel}>
          <ColorPicker
            hue={item.hue}
            onChange={(nextHue) => classActions.updateHue(categoryKey, item.key, nextHue)}
          />
        </div>
      )}
    </>
  );
};

interface CategoryItemProps {
  category: CategoryState;
  error?: string;
  fieldErrors: Record<string, string>;
  activeColorKey: string | null;
  toggleColorPicker: (key: string) => void;
  closeColorPicker: () => void;
  categoryActions: CategoryActions;
  classActions: ClassActions;
}

const CategoryItem = ({
  category,
  error,
  fieldErrors,
  activeColorKey,
  toggleColorPicker,
  closeColorPicker,
  categoryActions,
  classActions,
}: CategoryItemProps) => {
  const accentHue = category.segmentClasses[0]?.hue ?? 210;

  return (
    <div
      className={clsx(s.group, category.collapsed && s.collapsed)}
      style={{ "--group-hue": accentHue } as CSSProperties}
    >
      <div className={s.groupRow}>
        <ChevronRight
          className={clsx(s.chevron, !category.collapsed && s.expanded)}
          size={13}
          onClick={() => categoryActions.toggle(category.key)}
        />

        <input
          className={clsx(s.nameInput, error && s.nameInputError)}
          value={category.name}
          placeholder="Название категории"
          onChange={(e) => categoryActions.updateName(category.key, e.target.value)}
        />

        <button
          type="button"
          className={clsx(s.iconButton, s.addButton)}
          onClick={() => classActions.addToCategory(category.key)}
        >
          <Plus size={14} />
        </button>

        <button
          type="button"
          className={clsx(s.iconButton, s.removeButton)}
          onClick={() => {
            if (!window.confirm(`Удалить категорию "${category.name || "без названия"}"?`)) {
              return;
            }
            categoryActions.remove(category.key);
          }}
        >
          <X size={14} />
        </button>
      </div>

      {error ? <div className={s.inlineError}>{error}</div> : null}

      {!category.collapsed &&
        category.segmentClasses.map((item) => (
          <ClassRow
            key={item.key}
            item={item}
            categoryKey={category.key}
            error={fieldErrors[`class:${item.key}`]}
            isColorOpen={activeColorKey === item.key}
            toggleColorPicker={toggleColorPicker}
            closeColorPicker={closeColorPicker}
            classActions={classActions}
          />
        ))}
    </div>
  );
};

interface UngroupedBlockProps {
  items: ClassState[];
  fieldErrors: Record<string, string>;
  activeColorKey: string | null;
  toggleColorPicker: (key: string) => void;
  closeColorPicker: () => void;
  classActions: ClassActions;
}

const UngroupedBlock = ({
  items,
  fieldErrors,
  activeColorKey,
  toggleColorPicker,
  closeColorPicker,
  classActions,
}: UngroupedBlockProps) => {
  const accentHue = items[0]?.hue ?? 210;

  return (
    <div className={s.group} style={{ "--group-hue": accentHue } as CSSProperties}>
      <div className={s.groupRow}>
        <span className={s.nameInput}>Без категории</span>

        <button
          type="button"
          className={clsx(s.iconButton, s.addButton)}
          onClick={classActions.addUngrouped}
        >
          <Plus size={14} />
        </button>
      </div>

      {items.map((item) => (
        <ClassRow
          key={item.key}
          item={item}
          categoryKey={null}
          error={fieldErrors[`class:${item.key}`]}
          isColorOpen={activeColorKey === item.key}
          toggleColorPicker={toggleColorPicker}
          closeColorPicker={closeColorPicker}
          classActions={classActions}
        />
      ))}

      {items.length === 0 && (
        <div className={s.emptyClassHint}>
          Классов пока нет. Нажмите +, чтобы добавить класс без категории.
        </div>
      )}
    </div>
  );
};

export const ManageSegmentGroups = ({ group, compact, standardId, imageId }: Props) => (
  <Modal>
    <Modal.Trigger>
      {compact ? (
        <Button variant="ghost" size="sm">
          Edit classes
        </Button>
      ) : (
        <Button variant="ghost" size="sm">
          Manage classes
        </Button>
      )}
    </Modal.Trigger>

    <Modal.Content>
      <ManageSegmentGroupsModal group={group} standardId={standardId} imageId={imageId} />
    </Modal.Content>
  </Modal>
);

export const SegmentGroupsWorkspace = ({ group, standardId, imageId }: Props) => {
  const {
    categories,
    ungroupedClasses,
    saving,
    activeColorKey,
    toggleColorPicker,
    closeColorPicker,
    categoryActions,
    classActions,
    save,
    fieldErrors,
  } = useManageSegmentGroups(group, { standardId, imageId });

  return (
    <div className={s.workspace}>
      <div className={s.workspaceToolbar}>
        <div>
          <strong>Editable class catalog</strong>
          <span>{categories.length} categories · {ungroupedClasses.length} ungrouped classes</span>
        </div>

        <div className={s.workspaceActions}>
          <Button variant="ghost" size="sm" icon={Plus} onClick={categoryActions.add}>
            Category
          </Button>
          <Button variant="ghost" size="sm" icon={Plus} onClick={classActions.addUngrouped}>
            Class
          </Button>
        </div>
      </div>

      <div className={clsx(s.content, s.workspaceContent)}>
        <div className={clsx(s.list, s.workspaceList)}>
          {categories.map((category) => (
            <CategoryItem
              key={category.key}
              category={category}
              error={fieldErrors[`category:${category.key}`]}
              fieldErrors={fieldErrors}
              activeColorKey={activeColorKey}
              toggleColorPicker={toggleColorPicker}
              closeColorPicker={closeColorPicker}
              categoryActions={categoryActions}
              classActions={classActions}
            />
          ))}

          <UngroupedBlock
            items={ungroupedClasses}
            fieldErrors={fieldErrors}
            activeColorKey={activeColorKey}
            toggleColorPicker={toggleColorPicker}
            closeColorPicker={closeColorPicker}
            classActions={classActions}
          />
        </div>
      </div>

      <div className={s.workspaceFooter}>
        <span>Сначала добавь категории и классы, затем нажми Save. После сохранения они появятся в editor, Train и Inspect.</span>
        <Button disabled={saving} icon={Save} onClick={save}>
          Save classes
        </Button>
      </div>
    </div>
  );
};

const ManageSegmentGroupsModal = ({ group, standardId, imageId }: Props) => {
  const close = useModalClose();

  const {
    categories,
    ungroupedClasses,
    saving,
    activeColorKey,
    toggleColorPicker,
    closeColorPicker,
    categoryActions,
    classActions,
    save,
    fieldErrors,
  } = useManageSegmentGroups(group, { standardId, imageId });

  const handleSave = async () => {
    const ok = await save();
    if (ok) close();
  };

  return (
    <>
      <Modal.Header>Project classes</Modal.Header>

      <Modal.Body>
        <div className={s.content}>
          <div className={s.list}>
            {categories.map((category) => (
              <CategoryItem
                key={category.key}
                category={category}
                error={fieldErrors[`category:${category.key}`]}
                fieldErrors={fieldErrors}
                activeColorKey={activeColorKey}
                toggleColorPicker={toggleColorPicker}
                closeColorPicker={closeColorPicker}
                categoryActions={categoryActions}
                classActions={classActions}
              />
            ))}

            <UngroupedBlock
              items={ungroupedClasses}
              fieldErrors={fieldErrors}
              activeColorKey={activeColorKey}
              toggleColorPicker={toggleColorPicker}
              closeColorPicker={closeColorPicker}
              classActions={classActions}
            />
          </div>

          <Button variant="ghost" size="sm" icon={Plus} onClick={categoryActions.add} full>
            Добавить категорию
          </Button>
        </div>
      </Modal.Body>

      <Modal.Footer>
        <Button variant="ghost" onClick={close}>
          Отмена
        </Button>
        <Button disabled={saving} onClick={handleSave}>
          Сохранить
        </Button>
      </Modal.Footer>
    </>
  );
};
