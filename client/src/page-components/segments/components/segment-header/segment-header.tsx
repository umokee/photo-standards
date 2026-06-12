import { Badge } from "@/components/ui/badge/badge";
import Button from "@/components/ui/button/button";
import { ArrowLeft } from "lucide-react";
import s from "./segment-header.module.scss";

type Props = {
  standardName: string;
  currentIndex: number;
  totalImages: number;
  isReference: boolean;
  isCurrentAnnotated: boolean;
  segmentsCount: number;
  canGoPrev: boolean;
  canGoNext: boolean;
  hasNextUnannotated: boolean;
  onPrev: () => void;
  onNext: () => void;
  onNextUnannotated: () => void;
  onBack: () => void;
};

export const SegmentHeader = ({
  standardName,
  currentIndex,
  totalImages,
  isReference,
  isCurrentAnnotated,
  canGoPrev,
  canGoNext,
  hasNextUnannotated,
  onPrev,
  onNext,
  onNextUnannotated,
  onBack,
}: Props) => {
  return (
    <header className={s.root}>
      <div className={s.left}>
        <Button variant="ghost" size="sm" icon={ArrowLeft} onClick={onBack}>
          К эталону
        </Button>
        <div className={s.title}>{standardName}</div>
        {isReference && <Badge>Участвует в проверке</Badge>}
        <Badge type={isCurrentAnnotated ? "success" : "warning"}>
          {isCurrentAnnotated ? "Размечено" : "Не размечено"}
        </Badge>
      </div>

      <div className={s.actions}>
        <Button variant="ghost" size="sm" onClick={onPrev} disabled={!canGoPrev}>
          Пред.
        </Button>
        <span className={s.metaItem}>
          {currentIndex + 1} / {totalImages}
        </span>
        <Button variant="ghost" size="sm" onClick={onNext} disabled={!canGoNext}>
          След.
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onNextUnannotated}
          disabled={!hasNextUnannotated}
        >
          Неразмеч.
        </Button>
      </div>
    </header>
  );
};
