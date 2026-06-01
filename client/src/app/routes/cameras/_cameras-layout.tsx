import { Sidebar } from "@/components/layouts/sidebar/sidebar";
import { SplitLayout } from "@/components/layouts/split-layout/split-layout";
import Input from "@/components/ui/input/input";
import QueryState from "@/components/ui/query-state/query-state";
import useSidebar from "@/hooks/use-sidebar";
import { useGetCameras } from "@/page-components/cameras/api/get-cameras";
import { CreateCamera } from "@/page-components/cameras/components/create-camera";
import {
  filterCamerasBySearch,
  getCameraSidebarMeta,
  getCameraStatusMeta,
} from "@/page-components/cameras/lib/camera-view";
import { useCameraStatusLive } from "@/page-components/cameras/hooks/use-camera-status-live";
import { useMemo, useState } from "react";
import { Outlet, useNavigate, useParams } from "react-router-dom";
import { paths } from "../../paths";

export function Component() {
  const { cameraId = null } = useParams();
  const [search, setSearch] = useState("");

  const navigate = useNavigate();
  const { close: closeSidebar } = useSidebar();
  const { data: cameras = [], isLoading, isError } = useGetCameras();

  useCameraStatusLive({ scope: "cameras" });

  const filtered = useMemo(() => filterCamerasBySearch(cameras, search), [cameras, search]);

  return (
    <SplitLayout>
      <SplitLayout.Sidebar>
        <Sidebar>
          <Sidebar.Header>
            <Sidebar.HeaderTop>
              <Sidebar.Title>Камеры</Sidebar.Title>
            </Sidebar.HeaderTop>

            <Input placeholder="Поиск..." noMargin value={search} onChange={setSearch} />
          </Sidebar.Header>

          <Sidebar.List>
            <QueryState
              isLoading={isLoading}
              isError={isError}
              isEmpty={!filtered.length}
              emptyTitle="Нет камер"
              emptyDescription={search && "Попробуйте изменить запрос или очистить поиск"}
            >
              {filtered.map((camera) => {
                const status = getCameraStatusMeta(camera);

                return (
                  <Sidebar.Item
                    key={camera.id}
                    active={cameraId === camera.id}
                    onClick={() => {
                      navigate(paths.cameraDetail(camera.id));
                      closeSidebar();
                    }}
                  >
                    <Sidebar.ItemDot status={status.dotStatus} />

                    <Sidebar.ItemBody>
                      <Sidebar.ItemName>{camera.name}</Sidebar.ItemName>
                      <Sidebar.ItemMeta>{getCameraSidebarMeta(camera)}</Sidebar.ItemMeta>
                    </Sidebar.ItemBody>

                    <Sidebar.ItemSide>{camera.protocol.toUpperCase()}</Sidebar.ItemSide>
                  </Sidebar.Item>
                );
              })}
            </QueryState>
          </Sidebar.List>

          <Sidebar.Footer>
            <CreateCamera />
          </Sidebar.Footer>
        </Sidebar>
      </SplitLayout.Sidebar>

      <SplitLayout.Content>
        <SplitLayout.Body>
          <Outlet />
        </SplitLayout.Body>
      </SplitLayout.Content>
    </SplitLayout>
  );
}
