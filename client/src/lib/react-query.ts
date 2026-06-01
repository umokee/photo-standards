import { useNotificationStore } from "@/lib/notifications";
import { DefaultOptions, UseMutationOptions } from "@tanstack/react-query";
import { ApiError } from "./api-client";
import { getErrorMessage, isValidationError } from "./errors";

export const queryConfig = {
  queries: {
    refetchOnWindowFocus: false,
    retry: 1,
    staleTime: 1000 * 30,
  },
} satisfies DefaultOptions;

export type MutationConfig<MutationFnType extends (...args: any) => Promise<any>> =
  UseMutationOptions<Awaited<ReturnType<MutationFnType>>, ApiError, Parameters<MutationFnType>[0]>;

export const notifySuccess = (message: string) => {
  useNotificationStore.getState().addNotification({
    type: "success",
    message,
  });
};

export const notifyError = (message: string) => {
  useNotificationStore.getState().addNotification({
    type: "error",
    message,
  });
};

export const handleAppError = (error: unknown) => {
  if (isValidationError(error)) {
    notifyError(getErrorMessage(error));
    return;
  }

  notifyError(getErrorMessage(error));
};
