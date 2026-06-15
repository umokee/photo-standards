import type { Camera } from "@/types/contracts";
import { z } from "zod";

export type CameraProtocol = Camera["protocol"];

export type CameraPayloadBase = {
  name: string;
  description: string | null;
  protocol: CameraProtocol;
  host: string | null;
  port: number | null;
  path: string | null;
  stream_path: string | null;
  device_path: string | null;
  username: string | null;
  location: string | null;
  is_active: boolean;
  timeout_sec: number;
};

export type CameraFormValues = {
  name: string;
  description: string;
  protocol: CameraProtocol;
  host: string;
  port: string;
  stream_path: string;
  device_path: string;
  username: string;
  password: string;
  has_auth: boolean;
  location: string;
  timeout_sec: string;
  is_active: boolean;
};

export type BuildCameraPayloadResult<TData> =
  | { ok: true; data: TData }
  | { ok: false; errors: Record<string, string> };

export const cameraProtocolOptions = [
  { value: "rtsp", label: "RTSP" },
  { value: "http", label: "HTTP" },
  { value: "usb", label: "USB" },
] satisfies { value: CameraProtocol; label: string }[];

export const cameraFormFieldKeys = [
  "name",
  "description",
  "protocol",
  "host",
  "port",
  "stream_path",
  "device_path",
  "username",
  "password",
  "has_auth",
  "location",
  "timeout_sec",
  "is_active",
] as const satisfies readonly (keyof CameraFormValues)[];

export const initialCameraFormValues: CameraFormValues = {
  name: "",
  description: "",
  protocol: "rtsp",
  host: "",
  port: "554",
  stream_path: "",
  device_path: "",
  username: "",
  password: "",
  has_auth: false,
  location: "",
  timeout_sec: "5",
  is_active: true,
};

const cameraFormSchema = z
  .object({
    name: z.string().trim().min(1, "Укажите название"),
    description: z.string(),
    protocol: z.enum(["rtsp", "http", "usb"]),
    host: z.string(),
    port: z.string(),
    stream_path: z.string(),
    device_path: z.string(),
    username: z.string(),
    password: z.string(),
    has_auth: z.boolean(),
    location: z.string(),
    timeout_sec: z.string(),
    is_active: z.boolean(),
  })
  .superRefine((values, ctx) => {
    const timeout = Number(values.timeout_sec);
    if (!Number.isInteger(timeout) || timeout < 1 || timeout > 60) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["timeout_sec"],
        message: "Таймаут должен быть целым числом от 1 до 60",
      });
    }

    if (values.protocol === "usb") {
      if (!values.device_path.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["device_path"],
          message: "Выберите USB-камеру",
        });
      }

      return;
    }

    if (!values.host.trim()) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["host"],
        message: "Укажите IPv4-адрес камеры",
      });
    } else if (!isValidCameraHost(values.host)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["host"],
        message: "Укажите корректный IPv4-адрес",
      });
    }

    if (!values.stream_path.trim()) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["stream_path"],
        message: "Укажите путь потока",
      });
    } else if (!isValidStreamPath(values.stream_path)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["stream_path"],
        message: "Укажите путь потока без пробелов",
      });
    }

    if (values.port.trim()) {
      const port = Number(values.port);
      if (!Number.isInteger(port) || port < 1 || port > 65535) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["port"],
          message: "Укажите порт числом от 1 до 65535",
        });
      }
    }

    if (values.has_auth && !values.username.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["username"],
          message: "Укажите логин",
        });
    }
  });

type ParsedCameraFormValues = z.infer<typeof cameraFormSchema>;

export const applyCameraFormChange = <K extends keyof CameraFormValues>(
  values: CameraFormValues,
  key: K,
  value: CameraFormValues[K]
): CameraFormValues => {
  if (key === "protocol") {
    return applyCameraProtocolChange(values, value as CameraProtocol);
  }

  if (key === "has_auth" && value === false) {
    return {
      ...values,
      has_auth: false,
      username: "",
      password: "",
    };
  }

  return { ...values, [key]: value };
};

const applyCameraProtocolChange = (
  values: CameraFormValues,
  protocol: CameraProtocol
): CameraFormValues => {
  if (protocol === values.protocol) {
    return values;
  }

  if (protocol === "usb") {
    return {
      ...values,
      protocol,
      host: "",
      port: "",
      stream_path: "",
      username: "",
      password: "",
      has_auth: false,
    };
  }

  const defaultPort = getDefaultPort(protocol);
  const shouldResetPort =
    values.protocol === "usb" || values.port === "" || values.port === "554" || values.port === "80";

  return {
    ...values,
    protocol,
    port: shouldResetPort ? defaultPort : values.port,
    device_path: "",
  };
};

export const toCameraFormValues = (camera: Camera): CameraFormValues => ({
  name: camera.name,
  description: camera.description ?? "",
  protocol: camera.protocol,
  host: camera.host ?? "",
  port: camera.port != null ? String(camera.port) : getDefaultPort(camera.protocol),
  stream_path: camera.stream_path ?? camera.path ?? "",
  device_path: camera.device_path ?? "",
  username: camera.username ?? "",
  password: "",
  has_auth: Boolean(camera.username),
  location: camera.location ?? "",
  timeout_sec: String(camera.timeout_sec),
  is_active: camera.is_active,
});

export const parseCameraFormValues = (
  values: CameraFormValues
): BuildCameraPayloadResult<ParsedCameraFormValues> => {
  const parsed = cameraFormSchema.safeParse(values);

  if (!parsed.success) {
    return {
      ok: false,
      errors: Object.fromEntries(
        parsed.error.issues.map((issue) => [String(issue.path[0] ?? "form"), issue.message])
      ),
    };
  }

  return {
    ok: true,
    data: parsed.data,
  };
};

export const buildNormalizedCameraData = (
  values: ParsedCameraFormValues
): CameraPayloadBase => {
  const networkValues = values.protocol === "usb" ? null : buildNetworkValues(values);

  return {
    name: values.name.trim(),
    description: toNullable(values.description),
    protocol: values.protocol,
    host: networkValues?.host ?? null,
    port: networkValues?.port ?? null,
    path: null,
    stream_path: networkValues?.streamPath ?? null,
    device_path: values.protocol === "usb" ? toNullable(values.device_path) : null,
    username: networkValues?.username ?? null,
    location: toNullable(values.location),
    is_active: values.is_active,
    timeout_sec: Number(values.timeout_sec),
  };
};

export const buildComparableCameraData = (
  values: CameraFormValues
): CameraPayloadBase => {
  const timeout = Number(values.timeout_sec);

  if (values.protocol === "usb") {
    return {
      name: values.name.trim(),
      description: toNullable(values.description),
      protocol: values.protocol,
      host: null,
      port: null,
      path: null,
      stream_path: null,
      device_path: toNullable(values.device_path),
      username: null,
      location: toNullable(values.location),
      is_active: values.is_active,
      timeout_sec: Number.isInteger(timeout) && timeout >= 1 && timeout <= 60 ? timeout : 5,
    };
  }

  const rawPort = values.port.trim() ? Number(values.port) : Number(getDefaultPort(values.protocol));

  return {
    name: values.name.trim(),
    description: toNullable(values.description),
    protocol: values.protocol,
    host: toNullable(values.host),
    port: Number.isInteger(rawPort) && rawPort >= 1 && rawPort <= 65535 ? rawPort : null,
    path: null,
    stream_path: toNullable(values.stream_path),
    device_path: null,
    username: values.has_auth ? toNullable(values.username) : null,
    location: toNullable(values.location),
    is_active: values.is_active,
    timeout_sec: Number.isInteger(timeout) && timeout >= 1 && timeout <= 60 ? timeout : 5,
  };
};

const buildNetworkValues = (values: ParsedCameraFormValues) => {
  const port = values.port.trim() ? Number(values.port) : Number(getDefaultPort(values.protocol));

  return {
    host: values.host.trim(),
    port,
    streamPath: values.stream_path.trim(),
    username: values.has_auth ? toNullable(values.username) : null,
  };
};

const getDefaultPort = (protocol: CameraProtocol): string => {
  if (protocol === "http") {
    return "80";
  }

  if (protocol === "usb") {
    return "";
  }

  return "554";
};

const toNullable = (value: string): string | null => {
  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
};

const isValidCameraHost = (value: string): boolean => {
  const candidate = value.trim();

  if (!candidate || /\s/.test(candidate) || candidate.includes("://") || /[\\/@?#]/.test(candidate)) {
    return false;
  }

  const ipv4Parts = candidate.split(".");
  if (ipv4Parts.length !== 4) {
    return false;
  }

  return ipv4Parts.every((part) => /^\d{1,3}$/.test(part) && Number(part) >= 0 && Number(part) <= 255);
};

const isValidStreamPath = (value: string): boolean => {
  const candidate = value.trim();
  return candidate !== "" && !/\s/.test(candidate) && !candidate.includes("://") && !candidate.includes("@");
};
