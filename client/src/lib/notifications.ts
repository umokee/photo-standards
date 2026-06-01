import { create } from "zustand";

export type NotificationType = "success" | "error" | "warning" | "info";

export type Notification = {
  id: string;
  type: NotificationType;
  title?: string;
  message?: string;
  duration?: number;
};

type NotificationStore = {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, "id">) => void;
  dismissNotification: (id: string) => void;
};

const getNotificationDuration = (notification: Omit<Notification, "id">) => {
  return notification.duration ?? (notification.type === "error" ? 8000 : 5000);
};

export const useNotificationStore = create<NotificationStore>((set) => ({
  notifications: [],

  addNotification: (notification) => {
    const id = crypto.randomUUID();
    const duration = getNotificationDuration(notification);

    set((state) => ({
      notifications: [...state.notifications, { ...notification, id }].slice(-4),
    }));

    if (duration <= 0) {
      return;
    }

    window.setTimeout(() => {
      set((state) => ({
        notifications: state.notifications.filter((item) => item.id !== id),
      }));
    }, duration);
  },

  dismissNotification: (id) => {
    set((state) => ({
      notifications: state.notifications.filter((item) => item.id !== id),
    }));
  },
}));
