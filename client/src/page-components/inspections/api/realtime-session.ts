import { client, sendKeepaliveDelete } from "@/lib/api-client";
import { queryClient } from "@/lib/query-client";
import { queryKeys } from "@/lib/query-keys";
import { type MutationConfig } from "@/lib/react-query";
import type { InspectionRealtimeStatus } from "@/types/contracts";
import { queryOptions, useMutation, useQuery } from "@tanstack/react-query";

export const stopRealtimeSession = async (sessionId: string): Promise<void> => {
  await client.delete(`/yolo/inspection/realtime/sessions/${sessionId}`);
};

export const stopRealtimeSessionKeepalive = (sessionId: string): void => {
  sendKeepaliveDelete(`/yolo/inspection/realtime/sessions/${sessionId}`);
};

export const getRealtimeStatus = async (sessionId: string): Promise<InspectionRealtimeStatus> => {
  return client.get(`/yolo/inspection/realtime/sessions/${sessionId}/status`);
};

export const getRealtimeStatusQueryOptions = (sessionId: string | null) =>
  queryOptions({
    queryKey: queryKeys.inspections.realtimeStatus(sessionId),
    queryFn: () => getRealtimeStatus(sessionId!),
    enabled: Boolean(sessionId),
  });

type Options = {
  mutationConfig?: MutationConfig<typeof stopRealtimeSession>;
};

export const useStopRealtimeSession = ({ mutationConfig }: Options = {}) => {
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: stopRealtimeSession,
    onSuccess: (...args) => {
      const [, sessionId] = args;

      queryClient.removeQueries({
        queryKey: queryKeys.inspections.realtimeStatus(sessionId),
      });
      onSuccess?.(...args);
    },
    ...rest,
  });
};

export const useGetRealtimeStatus = (sessionId: string | null) => {
  return useQuery({
    ...getRealtimeStatusQueryOptions(sessionId),
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return state === "failed" ? false : 500;
    },
    refetchIntervalInBackground: false,
  });
};
