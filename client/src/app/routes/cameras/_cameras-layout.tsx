import { paths } from "@/app/paths";
import Input from "@/components/ui/input/input";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetCameras } from "@/page-components/cameras/api/get-cameras";
import { CreateCamera } from "@/page-components/cameras/components/create-camera";
import { filterCamerasBySearch, getCameraSidebarMeta, getCameraStatusMeta } from "@/page-components/cameras/lib/camera-view";
import { useCameraStatusLive } from "@/page-components/cameras/hooks/use-camera-status-live";
import clsx from "clsx";
import { useMemo, useState } from "react";
import { Link, Outlet, useParams } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { cameraId = null } = useParams();
  const [search, setSearch] = useState("");
  const { data: cameras = [], isLoading, isError } = useGetCameras();
  useCameraStatusLive({ scope: "cameras" });
  const filtered = useMemo(() => filterCamerasBySearch(cameras, search), [cameras, search]);

  return (
    <div className={p.page}>
      <header className={p.ultraHeader}><div><h1>Sources</h1><p>IP/USB камеры для snapshot и realtime deployment.</p></div><div className={p.headerActions}><CreateCamera /></div></header>
      <div className={p.cameraGrid}>
        <aside className={p.listPanel}>
          <div className={p.listPanelHeader}><Input noMargin placeholder="Search cameras..." value={search} onChange={setSearch} /></div>
          <div className={p.listItems}>
            <QueryState isLoading={isLoading} isError={isError} isEmpty={!filtered.length} emptyTitle="No cameras">
              {filtered.map((camera) => { const status = getCameraStatusMeta(camera); return <Link key={camera.id} className={clsx(p.listItem, cameraId === camera.id && p.listItemActive)} to={paths.cameraDetail(camera.id)}><span className={clsx(p.statusDot, status.dotStatus === "success" && p.statusOnline, status.dotStatus === "danger" && p.statusOffline)} /><span><strong>{camera.name}</strong><small>{getCameraSidebarMeta(camera)}</small></span><small>{camera.protocol.toUpperCase()}</small></Link>; })}
            </QueryState>
          </div>
        </aside>
        <Outlet />
      </div>
    </div>
  );
}
