import { Camera, RadioTower, ScanLine } from "lucide-react";
import p from "../platform-pages.module.scss";

export function Component() {
  return (
    <section className={p.sourceEmptyHero}>
      <div className={p.sourceEmptyIcon}><Camera /></div>
      <h3>Select a source</h3>
      <p>Выбери IP/USB камеру слева или создай новый источник для snapshot/realtime inspection.</p>
      <div className={p.sourceEmptySteps}>
        <span><Camera /> Add source</span>
        <span><ScanLine /> Test preview</span>
        <span><RadioTower /> Use in Deploy</span>
      </div>
    </section>
  );
}
