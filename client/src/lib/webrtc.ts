type ConnectWebRTCVideoParams = {
  offerUrl: string;
  video: HTMLVideoElement;
  fps?: number;
  connectTimeoutMs?: number;
  iceServers?: RTCIceServer[];
  localStream?: MediaStream;
  receiveVideo?: boolean;
  onDisconnected?: (message: string) => void;
};

const DEFAULT_ICE_SERVERS: RTCIceServer[] = [];

export async function connectWebRTCVideo({
  offerUrl,
  video,
  fps = 20,
  connectTimeoutMs = 20_000,
  iceServers = DEFAULT_ICE_SERVERS,
  localStream,
  receiveVideo = true,
  onDisconnected,
}: ConnectWebRTCVideoParams): Promise<() => void> {
  const pc = new RTCPeerConnection({ iceServers });

  let cleanedUp = false;
  let readySettled = false;
  let disconnectNotified = false;
  let readyTimeoutId: number | null = null;

  let resolveReady!: () => void;
  let rejectReady!: (error: Error) => void;

  const cleanup = () => {
    if (cleanedUp) return;
    cleanedUp = true;

    const stream = video.srcObject;
    if (stream instanceof MediaStream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    video.pause();
    video.srcObject = null;
    pc.close();
  };

  const notifyDisconnected = (message: string) => {
    if (cleanedUp || disconnectNotified) return;
    disconnectNotified = true;
    onDisconnected?.(message);
  };

  const readyPromise = new Promise<void>((resolve, reject) => {
    resolveReady = () => {
      if (readySettled) return;
      readySettled = true;

      if (readyTimeoutId !== null) {
        window.clearTimeout(readyTimeoutId);
        readyTimeoutId = null;
      }

      resolve();
    };

    rejectReady = (error: Error) => {
      if (readySettled) return;
      readySettled = true;

      if (readyTimeoutId !== null) {
        window.clearTimeout(readyTimeoutId);
        readyTimeoutId = null;
      }

      reject(error);
    };

    readyTimeoutId = window.setTimeout(() => {
      rejectReady(new Error("Не удалось дождаться видеопотока"));
    }, connectTimeoutMs);
  });

  const localVideoTracks = localStream?.getVideoTracks() ?? [];

  if (localVideoTracks.length > 0 && localStream) {
    localVideoTracks.forEach((track) => {
      pc.addTransceiver(track, {
        direction: receiveVideo ? "sendrecv" : "sendonly",
        streams: [localStream],
      });
    });
  } else if (receiveVideo) {
    pc.addTransceiver("video", { direction: "recvonly" });
  }

  pc.ontrack = (event) => {
    const stream = event.streams[0] ?? new MediaStream([event.track]);

    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    video.srcObject = stream;

    video.play().catch(() => {});
    resolveReady();
  };

  pc.onconnectionstatechange = () => {
    switch (pc.connectionState) {
      case "connected":
        resolveReady();
        return;
      case "failed":
        if (readySettled) {
          notifyDisconnected("Соединение завершилось с ошибкой");
        } else {
          rejectReady(new Error("Не удалось установить WebRTC-соединение"));
        }
        return;
      case "disconnected":
        if (readySettled) {
          notifyDisconnected("Соединение было прервано");
        } else {
          rejectReady(new Error("Соединение прервано до получения видеопотока"));
        }
        return;
      case "closed":
        if (readySettled) {
          notifyDisconnected("Соединение было закрыто");
        } else {
          rejectReady(new Error("Соединение было закрыто"));
        }
        return;
      default:
        return;
    }
  };

  pc.onicecandidateerror = (event) => {
    console.warn("WebRTC ICE candidate error:", event);
  };

  const offer = await pc.createOffer();

  await pc.setLocalDescription(offer);

  await waitForIceGatheringComplete(pc);

  const localDescription = pc.localDescription;
  if (!localDescription) {
    cleanup();
    throw new Error("Не удалось создать WebRTC offer");
  }

  const response = await fetch(offerUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      sdp: localDescription.sdp,
      type: localDescription.type,
      fps,
      receive_video: receiveVideo,
    }),
  });

  if (!response.ok) {
    const message = await readErrorMessage(response, "Не удалось подключить WebRTC поток");
    cleanup();
    throw new Error(message);
  }

  const answer = await response.json();

  await pc.setRemoteDescription({
    sdp: answer.sdp,
    type: answer.type,
  });

  try {
    await readyPromise;
  } catch (error) {
    cleanup();
    throw error;
  }

  return cleanup;
}

async function readErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }
  } catch {
    return fallback;
  }

  return fallback;
}

function waitForIceGatheringComplete(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === "complete") {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };

    pc.addEventListener("icegatheringstatechange", check);
  });
}
