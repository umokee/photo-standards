import { useGetCamera } from "@/page-components/cameras/api/get-camera";
import { CameraPreviewPanel } from "@/page-components/cameras/components/camera-preview-panel/camera-preview-panel";
import { DeleteCamera } from "@/page-components/cameras/components/delete-camera";
import { UpdateCamera } from "@/page-components/cameras/components/update-camera";
import { buildCameraDisplayUrl, formatCameraDateTime, formatCameraLastCheckedAgo, getCameraStatusMeta } from "@/page-components/cameras/lib/camera-view";
import clsx from "clsx";
import { Activity, Camera, CheckCircle2, Clock3, Link2, MapPin, Network, RadioTower, ShieldCheck, Timer, Wifi } from "lucide-react";
import { useLoaderData } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { cameraId } = useLoaderData() as { cameraId: string };
  const { data: camera } = useGetCamera(cameraId);
  const cameraUrl = buildCameraDisplayUrl(camera);
  const status = getCameraStatusMeta(camera);

  return (
    <section className={p.cameraDetailSurface}>
      <header className={p.cameraDetailHero}>
        <div className={p.sourceAvatar}><Camera /></div>
        <div>
          <span className={p.eyebrow}><RadioTower /> Source detail</span>
          <h3>{camera.name}</h3>
          <p>{camera.location || "No location"} · {camera.protocol.toUpperCase()} · {formatCameraLastCheckedAgo(camera.last_checked_at)}</p>
        </div>
        <span className={clsx(p.sourceStatusBadge, camera.last_status === "online" && p.sourceStatusOnline, camera.last_status === "offline" && p.sourceStatusOffline)}>{status.label}</span>
        <div className={p.headerActions}><UpdateCamera camera={camera} /><DeleteCamera id={camera.id} name={camera.name} /></div>
      </header>

      <div className={p.cameraDetailBody}>
        <section className={p.cameraPreviewCard}>
          <div className={p.previewCardHead}>
            <div><h3>Live preview</h3><p>Проверь поток перед использованием в Inspect.</p></div>
            <span>{camera.protocol.toUpperCase()}</span>
          </div>
          <CameraPreviewPanel camera={camera} />
        </section>

        <aside className={p.cameraDiagnosticsPanel}>
          <div className={p.sourceStatsGridCompact}>
            <SourceStat icon={ShieldCheck} label="Active" value={camera.is_active ? "Yes" : "No"} />
            <SourceStat icon={Activity} label="Status" value={camera.last_status} />
            <SourceStat icon={Timer} label="Timeout" value={`${camera.timeout_sec} sec`} />
            <SourceStat icon={Clock3} label="Checked" value={formatCameraDateTime(camera.last_checked_at)} />
          </div>

          <section className={p.detailInfoCard}>
            <h3>Connection</h3>
            <Info icon={Link2} label="URL" value={cameraUrl} />
            <Info icon={Network} label="Host" value={camera.host || "—"} />
            <Info icon={Wifi} label="Port" value={camera.port ?? "—"} />
            <Info icon={MapPin} label="Path" value={camera.path || camera.stream_path || camera.device_path || "—"} />
          </section>

          <section className={p.detailInfoCard}>
            <h3>Diagnostics</h3>
            <Info icon={Clock3} label="Last checked" value={formatCameraDateTime(camera.last_checked_at)} />
            <Info icon={CheckCircle2} label="Error" value={camera.last_error || "—"} />
            <Info icon={Camera} label="Description" value={camera.description || "—"} />
          </section>
        </aside>
      </div>
    </section>
  );
}

function SourceStat({ icon: Icon, label, value }: { icon: typeof Camera; label: string; value: string | number }) {
  return (
    <div className={p.sourceStatCard}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Info({ icon: Icon, label, value }: { icon: typeof Camera; label: string; value: string | number }) {
  return (
    <div className={p.connectionInfoRow}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
