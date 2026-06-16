import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import { DeleteGroup } from "@/page-components/groups/components/delete-group";
import { UpdateGroup } from "@/page-components/groups/components/update-group";
import { ManageSegmentGroups } from "@/page-components/segments/components/manage-segment-groups/manage-segment-groups";
import { CreateStandard } from "@/page-components/standards/components/create-standard";
import { GroupDetail } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { Box, Image, Tags } from "lucide-react";
import { Link, Outlet, useLoaderData, useOutletContext } from "react-router-dom";
import p from "../platform-pages.module.scss";

type GroupDetailOutletContext = { group: GroupDetail };

export const useGroupDetailOutletContext = () => useOutletContext<GroupDetailOutletContext>();

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: group } = useGetGroup(groupId);
  const referencePath = group.standards.find((standard) => standard.reference_path)?.reference_path;

  return (
    <div className={p.page}>
      <section className={p.datasetHeader}>
        {referencePath ? <img className={p.datasetThumb} src={`/storage/${referencePath}`} alt="" /> : <div className={p.datasetThumb} />}
        <div className={p.datasetTitle}>
          <h1>{group.name}</h1>
          <div className={p.metaLine}>
            <span><Image /> {group.stats.images_count} images</span>
            <span>{group.stats.annotated_images_count} labeled</span>
            <span><Box /> {group.stats.polygons_count} annotations</span>
            <span><Tags /> {group.stats.segment_classes_count} classes</span>
            <span>Created {formatDate(group.created_at)}</span>
          </div>
          <p>{group.description || "Dataset for standard-based visual inspection. Use it to manage reference views, annotations, models and inspection runs."}</p>
        </div>
        <div className={p.headerActions}>
          <CreateStandard groupId={group.id} />
          <ManageSegmentGroups group={group} />
          <UpdateGroup group={group} />
          <DeleteGroup id={group.id} name={group.name} />
        </div>
      </section>

      <nav className={p.tabs} aria-label="Dataset tabs">
        <Link className={p.active} to={paths.groupDetail(group.id)}>Images</Link>
        <Link to={paths.trainingGroup(group.id)}>Models</Link>
        <Link to={paths.inspectionGroup("photo", group.id)}>Deploy</Link>
        <Link to={paths.inspectionHistoryGroup(group.id)}>Runs</Link>
      </nav>

      <Outlet context={{ group }} />
    </div>
  );
}
