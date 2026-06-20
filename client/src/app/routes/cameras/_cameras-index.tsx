
import { EmptyStateCard } from "@/components/ui/empty-state-card/empty-state-card";
import { Camera, Network } from "lucide-react";
import s from "./_cameras-strict.module.scss";

export function Component() {
  return (
    <EmptyStateCard
      className={s.emptyState}
      icon={Camera}
      iconClassName={s.emptyIcon}
      eyebrow={<><Network /> Камеры</>}
      eyebrowClassName={s.eyebrow}
      title="Выбери источник изображения"
      description="Выбери источник и проверь preview, статус и подключение."
    />
  );
}
