export const DEVICE_CAMERA_ID = "__device_camera__";

export const DEVICE_CAMERA_OPTION = {
  value: DEVICE_CAMERA_ID,
  label: "Камера этого устройства",
};

export const isDeviceCameraId = (cameraId: string | null | undefined): boolean => {
  return cameraId === DEVICE_CAMERA_ID;
};

export const captureVideoFrameAsFile = async (
  video: HTMLVideoElement | null,
  filename = "device-camera.jpg"
): Promise<File> => {
  if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    throw new Error("Камера устройства ещё не отдала кадр");
  }

  const width = video.videoWidth;
  const height = video.videoHeight;

  if (!width || !height) {
    throw new Error("Не удалось определить размер кадра камеры");
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Не удалось подготовить кадр камеры");
  }

  ctx.drawImage(video, 0, 0, width, height);

  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", 0.85);
  });

  if (!blob) {
    throw new Error("Не удалось получить изображение с камеры");
  }

  return new File([blob], filename, { type: "image/jpeg" });
};
