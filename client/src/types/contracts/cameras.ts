export interface Camera {
  id: string;
  name: string;
  description: string | null;

  protocol: "rtsp" | "http" | "usb";
  host: string | null;
  port: number | null;
  path: string | null;
  stream_path: string | null;
  device_path: string | null;

  username: string | null;
  location: string | null;
  is_active: boolean;
  timeout_sec: number;

  last_status: "online" | "offline" | "unknown";
  last_checked_at: string | null;
  last_error: string | null;
  created_at: string;
}

export interface CameraTestResponse {
  status: "online" | "offline" | "unknown";
  message: string;
  checked_at: string;
  width: number | null;
  height: number | null;
}

export interface UsbCameraDevice {
  name: string;
  device_path: string;
  index: number | null;
  width: number | null;
  height: number | null;
  backend: string | null;
}
