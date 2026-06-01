import { client } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { UsbCameraDevice } from "@/types/contracts";
import { queryOptions, useQuery } from "@tanstack/react-query";

export const getUsbCameraDevices = (): Promise<UsbCameraDevice[]> => {
  return client.get("/cameras/usb-devices");
};

export const getUsbCameraDevicesQueryOptions = () => {
  return queryOptions({
    queryKey: queryKeys.cameras.usbDevices(),
    queryFn: getUsbCameraDevices,
  });
};

export const useGetUsbCameraDevices = (enabled: boolean) => {
  return useQuery({
    ...getUsbCameraDevicesQueryOptions(),
    enabled,
  });
};
