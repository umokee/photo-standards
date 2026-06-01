import { connectWebRTCVideo } from "@/lib/webrtc";

type ConnectCameraPreviewWebRTCParams = {
  cameraId: string;
  video: HTMLVideoElement;
  fps?: number;
  connectTimeoutMs?: number;
  onDisconnected?: (message: string) => void;
};

export function connectCameraPreviewWebRTC({
  cameraId,
  video,
  fps,
  connectTimeoutMs,
  onDisconnected,
}: ConnectCameraPreviewWebRTCParams): Promise<() => void> {
  return connectWebRTCVideo({
    offerUrl: `/api/cameras/${cameraId}/webrtc/offer`,
    video,
    fps,
    connectTimeoutMs,
    onDisconnected,
  });
}
