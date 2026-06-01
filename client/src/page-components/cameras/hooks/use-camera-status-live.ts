import { queryKeys } from "@/lib/query-keys";
import type { Camera } from "@/types/contracts";
import { QueryClient, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

type CameraStatusLiveScope = "cameras" | "inspection";

type CameraStatusLiveEvent =
  | {
      kind: "camera_status";
      event: "snapshot";
      cameras: Camera[];
    }
  | {
      kind: "camera_status";
      event: "camera";
      camera: Camera;
    };

type Options = {
  scope: CameraStatusLiveScope;
  enabled?: boolean;
};

export function useCameraStatusLive({ scope, enabled = true }: Options) {
  const qc = useQueryClient();

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/cameras/status?scope=${scope}`);

    ws.onmessage = (message) => {
      const data = JSON.parse(message.data) as CameraStatusLiveEvent;

      if (data.kind !== "camera_status") {
        return;
      }

      if (data.event === "snapshot") {
        applyCameraListToCache(qc, data.cameras);
        return;
      }

      if (data.event === "camera") {
        applyCameraToCache(qc, data.camera);
      }
    };

    ws.onerror = () => {
    };

    return () => {
      ws.close();
    };
  }, [enabled, qc, scope]);
}

function applyCameraListToCache(qc: QueryClient, cameras: Camera[]) {
  qc.setQueryData(queryKeys.cameras.all(), cameras);

  for (const camera of cameras) {
    qc.setQueryData(queryKeys.cameras.detail(camera.id), camera);
  }
}

function applyCameraToCache(qc: QueryClient, camera: Camera) {
  qc.setQueryData(queryKeys.cameras.detail(camera.id), camera);

  qc.setQueryData<Camera[] | undefined>(queryKeys.cameras.all(), (old) => {
    if (!old) {
      return [camera];
    }

    const exists = old.some((item) => item.id === camera.id);

    if (!exists) {
      return [camera, ...old];
    }

    return old.map((item) => (item.id === camera.id ? camera : item));
  });
}
