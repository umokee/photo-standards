import { paths } from "@/app/paths";
import Input from "@/components/ui/input/input";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetCameras } from "@/page-components/cameras/api/get-cameras";
import { CreateCamera } from "@/page-components/cameras/components/create-camera";
import { filterCamerasBySearch, getCameraSidebarMeta, getCameraStatusMeta } from "@/page-components/cameras/lib/camera-view";
import { useCameraStatusLive } from "@/page-components/cameras/hooks/use-camera-status-live";
import clsx from "clsx";
import { Camera, CircleDot, MonitorDot, RadioTower, Usb } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, Outlet, useParams } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { cameraId = null } = useParams();
  const [search, setSearch] = useState("");
  const { data: cameras = [], isLoading, isError } = useGetCameras();
  useCameraStatusLive({ scope: "cameras" });

  const filtered = useMemo(() => filterCamerasBySearch(cameras, search), [cameras, search]);
  const onlineCount = cameras.filter((camera) => camera.last_status === "online").length;
  const usbCount = cameras.filter((camera) => camera.protocol === "usb").length;
  const rtspCount = cameras.filter((camera) => camera.protocol === "rtsp").length;

  return (
    <div className={p.page}>
      <header className={p.cameraStudioHero}>
        <div>
          <span className={p.eyebrow}><Camera /> Camera control</span>
          <h1>Cameras</h1>
          <p>Управление IP/RTSP/USB источниками для snapshot и realtime-проверки.</p>
        </div>
        <div className={p.cameraHeroStats}>
          <span><CircleDot /> {onlineCount} online</span>
          <span><RadioTower /> {rtspCount} RTSP</span>
          <span><Usb /> {usbCount} USB</span>
        </div>
        <div className={p.headerActions}><CreateCamera /></div>
      </header>

      <div className={p.cameraStudioGrid}>
        <aside className={p.cameraRailPanel}>
          <div className={p.listPanelHeader}>
            <Input noMargin placeholder="Search cameras, RTSP, USB..." value={search} onChange={setSearch} />
          </div>
          <div className={p.cameraRailSummary}>
            <span><MonitorDot /> {filtered.length} sources</span>
            <span>{onlineCount}/{cameras.length} online</span>
          </div>
          <div className={p.cameraListItems}>
            <QueryState isLoading={isLoading} isError={isError} isEmpty={!filtered.length} emptyTitle="No cameras" emptyDescription="Добавь IP/RTSP или USB источник.">
              {filtered.map((camera) => {
                const status = getCameraStatusMeta(camera);
                return (
                  <Link key={camera.id} className={clsx(p.cameraListItem, cameraId === camera.id && p.cameraListItemActive)} to={paths.cameraDetail(camera.id)}>
                    <span className={clsx(p.statusDot, status.dotStatus === "success" && p.statusOnline, status.dotStatus === "danger" && p.statusOffline)} />
                    <span className={p.cameraProtocolBadge}>{camera.protocol.toUpperCase()}</span>
                    <span><strong>{camera.name}</strong><small>{getCameraSidebarMeta(camera)}</small></span>
                    <b>{status.label}</b>
                  </Link>
                );
              })}
            </QueryState>
          </div>
        </aside>
        <Outlet />
      </div>
    </div>
  );
}
