import Button from "@/components/ui/button/button";
import {
  GroupDetail,
  SegmentClass,
  SegmentClassCategory,
  SegmentClassWithPoints,
} from "@/types/contracts";
import clsx from "clsx";
import { Trash2 } from "lucide-react";
import { ManageSegmentGroups } from "../manage-segment-groups/manage-segment-groups";
import s from "./segment-panel.module.scss";

export type DrawKind = "polygon" | "sam";

interface Props {
  group?: GroupDetail;
  categories: SegmentClassCategory[];
  ungroupedClasses: SegmentClass[];
  imageSegmentClasses: SegmentClassWithPoints[];
  selectedSegmentClassId: string | null;
  selectedContourIndex: number | null;
  onSelectSegmentClass: (id: string) => void;
  onStartDraw: (kind: DrawKind) => void;
  onSelectContour: (index: number | null) => void;
  onDeleteContour: (segmentClassId: string, contourIndex: number) => void;
}

export const SegmentPanel = ({
  group,
  categories,
  ungroupedClasses,
  imageSegmentClasses,
  selectedSegmentClassId,
  selectedContourIndex,
  onSelectSegmentClass,
  onStartDraw,
  onSelectContour,
  onDeleteContour,
}: Props) => {
  const allClasses = [...categories.flatMap((c) => c.segment_classes), ...ungroupedClasses];

  const imageContours = imageSegmentClasses.flatMap((segmentClass) =>
    segmentClass.points.map((_, contourIndex) => ({
      key: `${segmentClass.id}-${contourIndex}`,
      segmentClassId: segmentClass.id,
      segmentClassName: segmentClass.name,
      contourIndex,
      hue: segmentClass.hue,
    }))
  );

  return (
    <div className={s.root}>
      <div className={clsx(s.section, s.classes)}>
        <div className={s.sectionHeader}>
          <span className={s.sectionTitle}>Классы</span>
          {group && <ManageSegmentGroups group={group} compact />}
        </div>

        <div className={s.classes}>
          {categories.map((category) => (
            <div key={category.id} className={s.groupBlock}>
              <span className={s.groupLabel}>{category.name}</span>

              {category.segment_classes.map((cls) => {
                const imageItem = imageSegmentClasses.find((c) => c.id === cls.id);
                const hasPoints = !!imageItem?.points.length;

                return (
                  <div
                    key={cls.id}
                    className={clsx(s.classItem, selectedSegmentClassId === cls.id && s.selected)}
                    onClick={() => onSelectSegmentClass(cls.id)}
                  >
                    <div
                      className={s.classColor}
                      style={{
                        background: `hsl(${cls.hue}, 70%, 50%)`,
                      }}
                    />
                    <span className={s.classLabel}>{cls.name}</span>
                    <span className={clsx(s.classCount, hasPoints && s.has)}>
                      {hasPoints ? `${imageItem?.points.length} пол.` : "-"}
                    </span>
                  </div>
                );
              })}
            </div>
          ))}

          {!!ungroupedClasses.length && (
            <div className={s.groupBlock}>
              <span className={s.groupLabel}>Без категории</span>

              {ungroupedClasses.map((segmentClass) => {
                const imageItem = imageSegmentClasses.find((item) => item.id === segmentClass.id);
                const hasPoints = !!imageItem?.points.length;

                return (
                  <div
                    key={segmentClass.id}
                    className={clsx(
                      s.classItem,
                      selectedSegmentClassId === segmentClass.id && s.selected
                    )}
                    onClick={() => onSelectSegmentClass(segmentClass.id)}
                  >
                    <div
                      className={s.classColor}
                      style={{
                        background: `hsl(${segmentClass.hue}, 70%, 50%)`,
                      }}
                    />
                    <span className={s.classLabel}>{segmentClass.name}</span>
                    <span className={clsx(s.classCount, hasPoints && s.has)}>
                      {hasPoints ? `${imageItem?.points.length} пол.` : "-"}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <div className={clsx(s.section, s.anns)}>
        <div className={s.sectionHeader}>
          <span className={s.sectionTitle}>Полигоны</span>
          <span className={s.sectionMeta}>
            {imageContours.length ? `${imageContours.length} пол.` : "-"}
          </span>
        </div>

        <div className={s.anns}>
          {!imageContours.length ? (
            <div className={s.annEmpty}>На изображении пока нет полигонов</div>
          ) : (
            <>
              {imageContours.map((item) => (
                <div
                  key={item.key}
                  className={clsx(
                    s.annItem,
                    selectedSegmentClassId === item.segmentClassId &&
                      selectedContourIndex === item.contourIndex &&
                      s.selected
                  )}
                  onClick={() => {
                    onSelectSegmentClass(item.segmentClassId);
                    onSelectContour(item.contourIndex);
                  }}
                >
                  <div
                    className={s.classColor}
                    style={{
                      background: `hsl(${item.hue}, 70%, 50%)`,
                    }}
                  />
                  <div className={s.annPolygon}>
                    {item.segmentClassName} {item.contourIndex + 1}
                  </div>
                  <div
                    className={s.annDelete}
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteContour(item.segmentClassId, item.contourIndex);
                    }}
                  >
                    <Trash2 size={11} />
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      </div>

      <div className={s.actions}>
        <Button
          variant="ghost"
          size="sm"
          disabled={!selectedSegmentClassId}
          onClick={() => onStartDraw("polygon")}
        >
          Полигон
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={!selectedSegmentClassId}
          onClick={() => onStartDraw("sam")}
        >
          SAM
        </Button>
      </div>
    </div>
  );
};
