
import { Camera, CheckCircle2, Network, RadioTower, ScanLine, Usb } from "lucide-react";
import s from "./_cameras-strict.module.scss";

export function Component() {
  return (
    <section className={s.emptyState}>
      <div className={s.emptyIcon}><Camera /></div>
      <span className={s.eyebrow}><Network /> Camera workspace</span>
      <h3>Выбери источник изображения</h3>
      <p>
        IP/USB камера нужна для snapshot и realtime проверки. Слева список источников,
        здесь — preview, статус и диагностика подключения.
      </p>

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
    </section>
  );
}
