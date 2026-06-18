
import { EmptyStateCard } from "@/components/ui/empty-state-card/empty-state-card";
import { Camera, CheckCircle2, Network, RadioTower, ScanLine, Usb } from "lucide-react";
import s from "./_cameras-strict.module.scss";

export function Component() {
  return (
    <EmptyStateCard
      className={s.emptyState}
      icon={Camera}
      iconClassName={s.emptyIcon}
      eyebrow={<><Network /> Camera workspace</>}
      eyebrowClassName={s.eyebrow}
      title="Выбери источник изображения"
      description="IP/USB камера нужна для snapshot и realtime проверки. Слева список источников, здесь — preview, статус и диагностика подключения."
    >
      <div className={s.emptySteps}>
        <span><Camera /> Add source</span>
        <span><ScanLine /> Test preview</span>
        <span><CheckCircle2 /> Use in Inspect</span>
      </div>

      <div className={s.sourceTypeGrid}>
        <div><RadioTower /><strong>IP / RTSP</strong><small>stream path, timeout, live status</small></div>
        <div><Usb /><strong>USB camera</strong><small>device path and local preview</small></div>
        <div><Network /><strong>Diagnostics</strong><small>last status, error and latency</small></div>
      </div>
    </EmptyStateCard>
  );
}
