import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { Link } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { data: groups } = useGetGroups();
  return (
    <div className={p.page}>
      <header className={p.ultraHeader}><div><h1>Runs</h1><p>История проверок по datasets.</p></div></header>
      <QueryState isEmpty={!groups.length} size="page" emptyTitle="No runs" emptyDescription="Проверки появятся после сохранения результатов.">
        <div className={p.gallery}>
          {groups.map((group) => <Link className={p.galleryCard} key={group.id} to={paths.inspectionHistoryGroup(group.id)}><div className={p.galleryImage}>{group.stats.inspections_count}</div><div className={p.galleryBody}><strong>{group.name}</strong><small>{group.stats.inspections_count} inspection runs</small></div></Link>)}
        </div>
      </QueryState>
    </div>
  );
}
