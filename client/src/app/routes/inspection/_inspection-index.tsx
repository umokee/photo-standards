import { Database, MousePointer2 } from "lucide-react";
import p from "../platform-pages.module.scss";

export function Component() {
  return (
    <div className={p.inspectEmptyStateV17}>
      <Database />
      <h2>Выбери dataset</h2>
      <p>Проверка начинается с изделия/dataset. После выбора появятся эталонные виды, источник изображения и кнопка запуска.</p>
      <span><MousePointer2 /> Используй верхнюю панель настройки.</span>
    </div>
  );
}
