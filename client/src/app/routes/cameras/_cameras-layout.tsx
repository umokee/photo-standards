
import { paths } from "@/app/paths";
import Input from "@/components/ui/input/input";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetCameras } from "@/page-components/cameras/api/get-cameras";
import { CreateCamera } from "@/page-components/cameras/components/create-camera";
import {
  filterCamerasBySearch,
  getCameraSidebarMeta,
  getCameraStatusMeta,
} from "@/page-components/cameras/lib/camera-view";
import { useCameraStatusLive } from "@/page-components/cameras/hooks/use-camera-status-live";
import clsx from "clsx";
import { Camera, CircleDot, MonitorDot, RadioTower, Usb } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, Outlet, useParams } from "react-router-dom";
import s from "./_cameras-strict.module.scss";

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
    <div className={s.page}>
      <header className={s.hero}>
        <div className={s.heroMain}>
          <span className={s.eyebrow}><Camera /> Камеры</span>
          <h1>Камеры</h1>
          <p>IP/RTSP/HTTP/USB источники для snapshot и realtime.</p>
        </div>

        <div className={s.heroStats}>
          <span><CircleDot /> {onlineCount} online</span>
          <span><RadioTower /> {rtspCount} RTSP</span>
          <span><Usb /> {usbCount} USB</span>
        </div>

        <div className={s.actions}><CreateCamera /></div>
      </header>

      <div className={s.workspace}>
        <aside className={s.rail}>
          <div className={s.railHeader}>
            <Input noMargin placeholder="Поиск камер..." value={search} onChange={setSearch} />
            <div className={s.railSummary}>
              <span><MonitorDot /> {filtered.length} sources</span>
              <span>{onlineCount}/{cameras.length} online</span>
            </div>
          </div>

          <div className={s.list}>
            <QueryState
              isLoading={isLoading}
              isError={isError}
              isEmpty={!filtered.length}
              emptyTitle="Нет камер"
              emptyDescription="Добавь IP/RTSP/HTTP или USB источник."
            >
              {filtered.map((camera) => {
                const status = getCameraStatusMeta(camera);
                const meta = getCameraSidebarMeta(camera).join(" · ");

                return (
                  <Link
                    key={camera.id}
                    className={clsx(s.item, cameraId === camera.id && s.itemActive)}
                    to={paths.cameraDetail(camera.id)}
                  >
                    <span
                      className={clsx(
                        s.statusDot,
                        status.dotStatus === "ok" && s.statusOk,
                        status.dotStatus === "warning" && s.statusWarning,
                        status.dotStatus === "error" && s.statusError,
                      )}
                    />
                    <span className={s.protocol}>{camera.protocol.toUpperCase()}</span>
                    <span className={s.itemBody}>
                      <strong>{camera.name}</strong>
                      <small>{meta || camera.description || "Без описания"}</small>
                    </span>
                    <b>{status.label}</b>
                  </Link>
                );
              })}
            </QueryState>
          </div>
        </aside>

        <main className={s.detailSlot}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
