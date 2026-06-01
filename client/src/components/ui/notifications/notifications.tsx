import { useNotificationStore } from "@/lib/notifications";
import { createPortal } from "react-dom";
import { Notification } from "./notification";
import s from "./notification.module.scss";

export function Notifications() {
  const notifications = useNotificationStore((state) => state.notifications);

  return createPortal(
    <div className={s.container}>
      {notifications.map((notification) => (
        <Notification key={notification.id} notification={notification} />
      ))}
    </div>,
    document.body
  );
}
