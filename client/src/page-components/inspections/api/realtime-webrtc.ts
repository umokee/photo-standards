import { connectWebRTCVideo } from "@/lib/webrtc";

type ConnectRealtimeWebRTCParams = {
  sessionId: string;
  video: HTMLVideoElement;
  fps?: number;
  connectTimeoutMs?: number;
  localStream?: MediaStream;
  receiveVideo?: boolean;
  onDisconnected?: (message: string) => void;
};

export function connectRealtimeInspectionWebRTC({
  sessionId,
  video,
  fps,
  connectTimeoutMs,
  localStream,
  receiveVideo,
  onDisconnected,
}: ConnectRealtimeWebRTCParams): Promise<() => void> {
  return connectWebRTCVideo({
    offerUrl: `/api/yolo/inspection/realtime/sessions/${sessionId}/webrtc/offer`,
    video,
    fps,
    connectTimeoutMs,
    localStream,
    receiveVideo,
    onDisconnected,
  });
}
