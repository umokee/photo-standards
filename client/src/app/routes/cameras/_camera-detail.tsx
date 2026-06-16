import { InfoRow } from "@/components/ui/info-row/info-row";
import SurfaceSection from "@/components/ui/surface-section/surface-section";
import { useGetCamera } from "@/page-components/cameras/api/get-camera";
import { CameraPreviewPanel } from "@/page-components/cameras/components/camera-preview-panel/camera-preview-panel";
import { DeleteCamera } from "@/page-components/cameras/components/delete-camera";
import { UpdateCamera } from "@/page-components/cameras/components/update-camera";
import { buildCameraDisplayUrl, formatCameraDateTime, formatCameraLastCheckedAgo } from "@/page-components/cameras/lib/camera-view";
import { useLoaderData } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { cameraId } = useLoaderData() as { cameraId: string };
  const { data: camera } = useGetCamera(cameraId);
  const cameraUrl = buildCameraDisplayUrl(camera);
  return (
    <section className={p.panelCard}>
      <div className={p.cardTitleRow}>
        <div><h3>{camera.name}</h3><p>{camera.location || "No location"} · {camera.protocol.toUpperCase()} · {formatCameraLastCheckedAgo(camera.last_checked_at)}</p></div>
        <div className={p.headerActions}><UpdateCamera camera={camera} /><DeleteCamera id={camera.id} name={camera.name} /></div>
      </div>
      <CameraPreviewPanel camera={camera} />
      <div className={p.grid2}>
        <SurfaceSection title="Main" transparent><InfoRow label="Active" value={camera.is_active ? "Yes" : "No"} /><InfoRow label="Status" value={camera.last_status} /><InfoRow label="Timeout" value={`${camera.timeout_sec} sec`} /></SurfaceSection>
        <SurfaceSection title="Connection" transparent><InfoRow label="URL" value={cameraUrl} valueWrap="wrap" /><InfoRow label="Last checked" value={formatCameraDateTime(camera.last_checked_at)} /><InfoRow label="Error" value={camera.last_error || "—"} /></SurfaceSection>
      </div>
    </section>
  );
}
