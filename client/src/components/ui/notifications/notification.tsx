import clsx from "clsx";
import { X } from "lucide-react";
import { Notification as NotificationData, useNotificationStore } from "../../../lib/notifications";
import s from "./notification.module.scss";

type Props = {
  notification: NotificationData;
};

export function Notification({ notification }: Props) {
  const dismiss = useNotificationStore((state) => state.dismissNotification);
  const duration = notification.duration ?? (notification.type === "error" ? 8000 : 5000);

  return (
    <div
      className={clsx(s.root, s[notification.type])}
      style={{ "--duration": `${duration}ms` } as React.CSSProperties}
    >
      <div className={s.content}>
        <span className={s.title}>{notification.title}</span>
        {notification.message && <span className={s.message}>{notification.message}</span>}
      </div>
      <button type="button" className={s.close} onClick={() => dismiss(notification.id)}>
        <X size={13} strokeWidth={1.75} />
      </button>
    </div>
  );
}
