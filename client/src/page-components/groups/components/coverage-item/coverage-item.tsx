import { GroupDetail } from "@/types/contracts";

import ProgressBar from "@/components/ui/progress-bar/progress-bar";
import s from "./coverage-item.module.scss";

type Props = {
  standard: GroupDetail["standards"][0];
};

export const CoverageItem = ({ standard }: Props) => {
  const done = standard.annotated_images_count >= standard.images_count;

  return (
    <div className={s.root}>
      <div className={s.head}>
        <span className={s.name}>{standard.name}</span>
        <strong className={s.count}>
          {standard.annotated_images_count} / {standard.images_count}
        </strong>
      </div>

      <ProgressBar
        value={standard.annotated_images_count}
        max={standard.images_count}
        warn={!done}
        showLabel={false}
      />
    </div>
  );
};
