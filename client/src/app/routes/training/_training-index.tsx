import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { FolderOpen } from "lucide-react";
import { Link } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { data: groups } = useGetGroups();

  return (
    <div className={p.page}>
      <header className={p.ultraHeader}>
        <div><h1>Projects</h1><p>Обучение YOLO-моделей и импорт весов привязаны к dataset изделия.</p></div>
      </header>
      <QueryState isEmpty={!groups.length} size="page" emptyTitle="No projects" emptyDescription="Сначала создай dataset в Annotate.">
        <div className={p.gallery}>
          {groups.map((group) => (
            <Link className={p.galleryCard} key={group.id} to={paths.trainingGroup(group.id)}>
              <div className={p.galleryImage}><FolderOpen /></div>
              <div className={p.galleryBody}>
                <strong>{group.name}</strong>
                <small>{group.stats.models_count} models · {group.stats.images_count} images · {group.stats.segment_classes_count} classes</small>
              </div>
            </Link>
          ))}
        </div>
      </QueryState>
    </div>
  );
}
