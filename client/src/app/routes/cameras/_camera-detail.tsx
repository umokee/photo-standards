import { InfoRow } from "@/components/ui/info-row/info-row";
import SurfaceSection from "@/components/ui/surface-section/surface-section";
import { useGetCamera } from "@/page-components/cameras/api/get-camera";
import { CameraPreviewPanel } from "@/page-components/cameras/components/camera-preview-panel/camera-preview-panel";
import { DeleteCamera } from "@/page-components/cameras/components/delete-camera";
import { UpdateCamera } from "@/page-components/cameras/components/update-camera";
import { buildCameraDisplayUrl, formatCameraDateTime, formatCameraLastCheckedAgo, getCameraStatusMeta } from "@/page-components/cameras/lib/camera-view";
import clsx from "clsx";
import { Activity, Camera, Clock3, RadioTower, ShieldCheck, Timer } from "lucide-react";
import { useLoaderData } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { cameraId } = useLoaderData() as { cameraId: string };
  const { data: camera } = useGetCamera(cameraId);
  const cameraUrl = buildCameraDisplayUrl(camera);
  const status = getCameraStatusMeta(camera);

  return (
    <section className={p.sourceDetailShell}>
      <div className={p.sourceDetailHeader}>
        <div className={p.sourceAvatar}><Camera /></div>
        <div>
          <span className={p.eyebrow}><RadioTower /> Source</span>
          <h3>{camera.name}</h3>
          <p>{camera.location || "No location"} · {camera.protocol.toUpperCase()} · {formatCameraLastCheckedAgo(camera.last_checked_at)}</p>
        </div>
        <span className={clsx(p.sourceStatusBadge, camera.last_status === "online" && p.sourceStatusOnline, camera.last_status === "offline" && p.sourceStatusOffline)}>{status.label}</span>
        <div className={p.headerActions}><UpdateCamera camera={camera} /><DeleteCamera id={camera.id} name={camera.name} /></div>
      </div>

      <div className={p.sourcePreviewFrame}>
        <CameraPreviewPanel camera={camera} />
      </div>

      <div className={p.sourceStatsGrid}>
        <SourceStat icon={ShieldCheck} label="Active" value={camera.is_active ? "Yes" : "No"} />
        <SourceStat icon={Activity} label="Status" value={camera.last_status} />
        <SourceStat icon={Timer} label="Timeout" value={`${camera.timeout_sec} sec`} />
        <SourceStat icon={Clock3} label="Checked" value={formatCameraDateTime(camera.last_checked_at)} />
      </div>

      <div className={p.grid2}>
        <SurfaceSection title="Connection" transparent>
          <InfoRow label="URL" value={cameraUrl} valueWrap="wrap" />
          <InfoRow label="Host" value={camera.host || "—"} />
          <InfoRow label="Port" value={camera.port ?? "—"} />
          <InfoRow label="Path" value={camera.path || camera.stream_path || camera.device_path || "—"} valueWrap="wrap" />
        </SurfaceSection>
        <SurfaceSection title="Diagnostics" transparent>
          <InfoRow label="Last checked" value={formatCameraDateTime(camera.last_checked_at)} />
          <InfoRow label="Error" value={camera.last_error || "—"} valueWrap="wrap" />
          <InfoRow label="Description" value={camera.description || "—"} valueWrap="wrap" />
        </SurfaceSection>
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
