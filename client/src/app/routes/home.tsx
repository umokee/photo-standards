import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { formatDate } from "@/utils/formatDate";
import { Box, Database, FolderOpen, Image, Plus, Rocket, Upload, type LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import p from "./platform-pages.module.scss";

export function Component() {
  const { data: groups } = useGetGroups();
  const totals = groups.reduce(
    (acc, group) => ({
      standards: acc.standards + group.stats.standards_count,
      images: acc.images + group.stats.images_count,
      annotations: acc.annotations + group.stats.polygons_count,
      models: acc.models + group.stats.models_count,
      inspections: acc.inspections + group.stats.inspections_count,
    }),
    { standards: 0, images: 0, annotations: 0, models: 0, inspections: 0 }
  );

  const recent = [...groups]
    .sort((a, b) => +new Date(b.created_at) - +new Date(a.created_at))
    .slice(0, 8);

  return (
    <div className={p.page}>
      <header className={p.ultraHeader}>
        <div>
          <h1>Home</h1>
          <p>Панель контроля качества: эталоны, модели, проверки и камеры в одном workflow.</p>
        </div>
      </header>

      <section className={p.homeHeroGrid}>
        <div className={p.welcomeCard}>
          <div className={p.userRow}>
            <span className={p.bigAvatar}>Q</span>
            <div>
              <h2>Welcome back, operator</h2>
              <p>@local · visual inspection workspace</p>
            </div>
            <span className={p.planBadge}>Local</span>
          </div>

          <div className={p.statStrip}>
            <Stat icon={Database} value={groups.length} label="Projects" />
            <Stat icon={Image} value={totals.images} label="Images" />
            <Stat icon={Box} value={totals.annotations} label="Annotations" />
            <Stat icon={FolderOpen} value={totals.standards} label="Standards" />
            <Stat icon={Rocket} value={totals.inspections} label="Runs" />
          </div>
        </div>

        <div className={p.quickVideoCard}>
          <div className={p.videoMock}>
            <div className={p.videoDevice}>
              <span />
              <span />
              <b>MISSING</b>
            </div>
          </div>
          <div className={p.chips}><span>YOLO</span><span>LightGlue</span><span>Homography</span></div>
        </div>
      </section>

      <section className={p.homeCards}>
        <div className={p.workspaceCard}>
          <div className={p.cardHead}>
            <div><Database /><h3>Datasets</h3><p>Группы изделий, эталонные кадры и полигоны.</p></div>
            <Link to={paths.groups()}><Button icon={Plus} size="sm">New Dataset</Button></Link>
          </div>
          <div className={p.dropzonePreview}><Upload /><span>Drop reference images or standards</span><small>Images · masks · polygons · reference views</small></div>
          <div className={p.miniDataset}>{groups[0]?.name ?? "No datasets yet"}<small>{totals.images} images · {totals.annotations} polygons</small></div>
          <Link className={p.viewAll} to={paths.groups()}>View all →</Link>
        </div>

        <div className={p.workspaceCard}>
          <div className={p.cardHead}>
            <div><FolderOpen /><h3>Projects</h3><p>Обучение моделей и эксперименты.</p></div>
            <Link to={paths.training()}><Button icon={Plus} size="sm">New Model</Button></Link>
          </div>
          <div className={p.dropzonePreview}><Upload /><span>Drop .pt model files</span><small>YOLO weights and training runs</small></div>
          {groups.slice(0, 2).map((group) => (
            <Link key={group.id} className={p.projectRow} to={paths.trainingGroup(group.id)}>
              <span>{group.name.slice(0, 1).toUpperCase()}</span>
              <strong>{group.name}</strong>
              <small>{group.stats.models_count} model</small>
            </Link>
          ))}
          <Link className={p.viewAll} to={paths.training()}>View all →</Link>
        </div>

        <div className={p.storageCard}>
          <h3>Storage</h3>
          <p>{Math.max(1, Math.round(totals.images * 0.35))} MB / local workspace</p>
          <div className={p.storageBar}><span style={{ width: `${Math.min(100, Math.max(4, totals.images * 3))}%` }} /></div>
          <ul>
            <li><span>Projects</span><b>{groups.length}</b></li>
            <li><span>Standards</span><b>{totals.standards}</b></li>
            <li><span>Models</span><b>{totals.models}</b></li>
            <li><span>Images</span><b>{totals.images}</b></li>
            <li><span>Inspection runs</span><b>{totals.inspections}</b></li>
          </ul>
        </div>
      </section>

      <section className={p.activityCard}>
        <div className={p.cardTitleRow}><div><h3>Recent Activity</h3><p>Последние datasets, models и inspection runs.</p></div></div>
        <div className={p.tableLike}>
          <div className={p.tableHead}><span>Name</span><span>Type</span><span>Description</span><span>Status</span><span>Updated</span></div>
          {recent.map((group) => (
            <Link className={p.tableRow} key={group.id} to={paths.groupDetail(group.id)}>
              <span>{group.name}</span><span>Dataset</span><span>{group.stats.images_count} images · {group.stats.polygons_count} polygons</span><span><b className={p.ok}>Ready</b></span><span>{formatDate(group.created_at)}</span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}

function Stat({ icon: Icon, value, label }: { icon: LucideIcon; value: number; label: string }) {
  return <div className={p.statBox}><Icon /><strong>{value}</strong><span>{label}</span></div>;
}
