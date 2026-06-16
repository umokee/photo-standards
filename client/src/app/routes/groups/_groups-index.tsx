import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { CreateGroup } from "@/page-components/groups/components/create-group";
import { formatDate } from "@/utils/formatDate";
import { Database, Image } from "lucide-react";
import { Link } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { data: groups } = useGetGroups();

  return (
    <div className={p.page}>
      <header className={p.ultraHeader}>
        <div>
          <h1>Datasets</h1>
          <p>Здесь группы изделий работают как datasets: изображения, эталоны, классы и полигоны.</p>
        </div>
        <div className={p.headerActions}><CreateGroup /></div>
      </header>

      <QueryState isEmpty={!groups.length} size="page" emptyTitle="Нет datasets" emptyDescription="Создай первый проект изделия для эталонов.">
        <div className={p.gallery}>
          {groups.map((group) => (
            <Link className={p.galleryCard} key={group.id} to={paths.groupDetail(group.id)}>
              <div className={p.galleryImage}><Database /></div>
              <div className={p.galleryBody}>
                <strong>{group.name}</strong>
                <small>{group.description || "Dataset for visual quality control"}</small>
                <div className={p.metaLine}>
                  <span><Image /> {group.stats.images_count} images</span>
                  <span>{group.stats.polygons_count} annotations</span>
                  <span>{formatDate(group.created_at)}</span>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </QueryState>
    </div>
  );
}
