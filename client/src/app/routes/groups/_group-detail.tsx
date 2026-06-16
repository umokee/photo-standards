import { paths } from "@/app/paths";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import { DeleteGroup } from "@/page-components/groups/components/delete-group";
import { UpdateGroup } from "@/page-components/groups/components/update-group";
import { ManageSegmentGroups } from "@/page-components/segments/components/manage-segment-groups/manage-segment-groups";
import { CreateStandard } from "@/page-components/standards/components/create-standard";
import { GroupDetail } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import clsx from "clsx";
import { Box, CheckCircle2, Database, Image, Layers3, Rocket, Tags } from "lucide-react";
import { Link, Outlet, useLoaderData, useLocation, useOutletContext } from "react-router-dom";
import p from "../platform-pages.module.scss";

type GroupDetailOutletContext = { group: GroupDetail };

export const useGroupDetailOutletContext = () => useOutletContext<GroupDetailOutletContext>();

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const { data: group } = useGetGroup(groupId);
  const location = useLocation();
  const referencePath = group.standards.find((standard) => standard.reference_path)?.reference_path;
  const labeledPercent = group.stats.images_count
    ? Math.round((group.stats.annotated_images_count / group.stats.images_count) * 100)
    : 0;
  const coveragePercent = group.stats.segment_classes_count
    ? Math.min(100, Math.round((group.stats.polygons_count / Math.max(1, group.stats.segment_classes_count)) * 10))
    : 0;

  return (
    <div className={p.page}>
      <section className={p.datasetHeroCard}>
        <div className={p.datasetHeroMedia}>
          {referencePath ? <img src={`/storage/${referencePath}`} alt="" /> : <Database />}
          <span className={p.datasetHeroBadge}>{group.stats.standards_count} views</span>
        </div>

        <div className={p.datasetHeroBody}>
          <div className={p.datasetBreadcrumbLine}>Annotate <span>/</span> Dataset</div>
          <h1>{group.name}</h1>
          <p>{group.description || "Dataset for standard-based visual inspection. Manage reference views, annotations, models and inspection runs."}</p>
          <div className={p.metaLine}>
            <span><Image /> {group.stats.images_count} images</span>
            <span><CheckCircle2 /> {group.stats.annotated_images_count} labeled</span>
            <span><Box /> {group.stats.polygons_count} polygons</span>
            <span><Tags /> {group.stats.segment_classes_count} classes</span>
            <span>Created {formatDate(group.created_at)}</span>
          </div>
        </div>

        <div className={p.datasetHeroActions}>
          <CreateStandard groupId={group.id} />
          <ManageSegmentGroups group={group} />
          <UpdateGroup group={group} />
          <DeleteGroup id={group.id} name={group.name} />
        </div>
      </section>

      <section className={p.datasetQualityStrip}>
        <QualityCard icon={Image} label="Image labeling" value={`${labeledPercent}%`} progress={labeledPercent} text={`${group.stats.annotated_images_count}/${group.stats.images_count} images labeled`} />
        <QualityCard icon={Layers3} label="Polygon density" value={String(group.stats.polygons_count)} progress={coveragePercent} text={`${group.stats.segment_classes_count} classes configured`} />
        <QualityCard icon={Rocket} label="Inspection runs" value={String(group.stats.inspections_count)} progress={group.stats.inspections_count ? 100 : 8} text="Deploy checks saved in history" />
      </section>

      <nav className={p.tabs} aria-label="Dataset tabs">
        <Link className={clsx(location.pathname.includes("/standards") || location.pathname.endsWith(group.id) ? p.active : undefined)} to={paths.groupDetail(group.id)}>Images</Link>
        <Link to={paths.trainingGroup(group.id)}>Models</Link>
        <Link to={paths.inspectionGroup("photo", group.id)}>Deploy</Link>
        <Link to={paths.inspectionHistoryGroup(group.id)}>Runs</Link>
      </nav>

      <Outlet context={{ group }} />
    </div>
  );
}

function QualityCard({ icon: Icon, label, value, text, progress }: { icon: typeof Image; label: string; value: string; text: string; progress: number }) {
  return (
    <div className={p.qualityCard}>
      <Icon />
      <div>
        <span>{label}</span>
        <b>{value}</b>
        <small>{text}</small>
        <div className={p.referenceProgress}><span style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div>
      </div>
    </div>
  );
}
