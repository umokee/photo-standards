import { Camera, CheckCircle2, Network, RadioTower, ScanLine, Usb } from "lucide-react";
import p from "../platform-pages.module.scss";

export function Component() {
  return (
    <section className={p.sourceEmptyHero}>
      <div className={p.sourceEmptyIcon}><Camera /></div>
      <span className={p.eyebrow}><Network /> Camera workspace</span>
      <h3>Выбери источник изображения</h3>
      <p>IP/USB камера нужна для snapshot и realtime проверки. Слева появится список источников, а здесь — preview, статус и диагностика подключения.</p>
      <div className={p.sourceEmptySteps}>
        <span><Camera /> Add source</span>
        <span><ScanLine /> Test preview</span>
        <span><CheckCircle2 /> Use in Inspect</span>
      </div>
      <div className={p.sourceTypeGrid}>
        <div><RadioTower /><strong>IP / RTSP</strong><small>stream path, timeout, live status</small></div>
        <div><Usb /><strong>USB camera</strong><small>device path and local preview</small></div>
        <div><Network /><strong>Diagnostics</strong><small>last status, error and latency</small></div>
      </div>
    </section>
  );
}
