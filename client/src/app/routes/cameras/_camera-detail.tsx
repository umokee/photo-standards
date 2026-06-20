
import { useGetCamera } from "@/page-components/cameras/api/get-camera";
import { CameraPreviewPanel } from "@/page-components/cameras/components/camera-preview-panel/camera-preview-panel";
import { DeleteCamera } from "@/page-components/cameras/components/delete-camera";
import { UpdateCamera } from "@/page-components/cameras/components/update-camera";
import {
  buildCameraDisplayUrl,
  formatCameraDateTime,
  formatCameraLastCheckedAgo,
  getCameraStatusMeta,
} from "@/page-components/cameras/lib/camera-view";
import type { LucideIcon } from "lucide-react";
import clsx from "clsx";
import {
  Activity,
  Camera,
  CheckCircle2,
  Clock3,
  Link2,
  MapPin,
  Network,
  RadioTower,
  ShieldCheck,
  Timer,
  Wifi,
} from "lucide-react";
import { useLoaderData } from "react-router-dom";
import s from "./_cameras-strict.module.scss";

export function Component() {
  const { cameraId } = useLoaderData() as { cameraId: string };
  const { data: camera } = useGetCamera(cameraId);

  if (!camera) {
    return null;
  }

  const cameraUrl = buildCameraDisplayUrl(camera);
  const status = getCameraStatusMeta(camera);

  return (
    <section className={s.detailSurface}>
      <header className={s.detailHeader}>
        <div className={s.sourceAvatar}><Camera /></div>
        <div className={s.detailTitle}>
          <span className={s.eyebrow}><RadioTower /> Source detail</span>
          <h3>{camera.name}</h3>
          <p>{camera.location || "No location"} · {camera.protocol.toUpperCase()} · {formatCameraLastCheckedAgo(camera.last_checked_at)}</p>
        </div>
        <span
          className={clsx(
            s.statusBadge,
            status.dotStatus === "ok" && s.statusBadgeOk,
            status.dotStatus === "warning" && s.statusBadgeWarning,
            status.dotStatus === "error" && s.statusBadgeError,
          )}
        >
          {status.label}
        </span>
        <div className={s.actions}><UpdateCamera camera={camera} triggerClassName={s.detailSecondaryAction} /><DeleteCamera id={camera.id} name={camera.name} triggerClassName={s.detailDangerAction} /></div>
      </header>

      <div className={s.detailBody}>
        <section className={s.previewCard}>
          <div className={s.cardHead}>
            <div><h3>Preview</h3><p>Проверь поток перед проверкой.</p></div>
            <span>{camera.protocol.toUpperCase()}</span>
          </div>
          <CameraPreviewPanel camera={camera} />
        </section>

        <aside className={s.diagnostics}>
          <div className={s.statGrid}>
            <SourceStat icon={ShieldCheck} label="Активна" value={camera.is_active ? "Yes" : "No"} />
            <SourceStat icon={Activity} label="Status" value={camera.last_status} />
            <SourceStat icon={Timer} label="Timeout" value={`${camera.timeout_sec} sec`} />
            <SourceStat icon={Clock3} label="Проверено" value={formatCameraDateTime(camera.last_checked_at)} />
          </div>

          <section className={s.infoCard}>
            <h3>Connection</h3>
            <Info icon={Link2} label="URL" value={cameraUrl} />
            <Info icon={Network} label="Host" value={camera.host || "—"} />
            <Info icon={Wifi} label="Port" value={camera.port ?? "—"} />
            <Info icon={MapPin} label="Path" value={camera.path || camera.stream_path || camera.device_path || "—"} />
          </section>

          <section className={s.infoCard}>
            <h3>Диагностика</h3>
            <Info icon={Clock3} label="Последняя проверка" value={formatCameraDateTime(camera.last_checked_at)} />
            <Info icon={CheckCircle2} label="Ошибка" value={camera.last_error || "—"} />
            <Info icon={Camera} label="Description" value={camera.description || "—"} />
          </section>
        </aside>
      </div>
    </section>
  );
}

function SourceStat({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string | number }) {
  return (
    <div className={s.statCard}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Info({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string | number }) {
  return (
    <div className={s.infoRow}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
