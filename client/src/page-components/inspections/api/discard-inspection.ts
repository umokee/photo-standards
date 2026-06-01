import { client, sendKeepaliveDelete } from "@/lib/api-client";
import { queryClient } from "@/lib/query-client";
import { type MutationConfig } from "@/lib/react-query";
import { getTaskQueryOptions } from "@/page-components/tasks/api/get-task";
import { useMutation } from "@tanstack/react-query";

export const discardInspection = async ({ taskId }: { taskId: string }): Promise<void> => {
  await client.delete(`/yolo/inspection/task/${taskId}`);
};

export const discardInspectionKeepalive = (taskId: string): void => {
  sendKeepaliveDelete(`/yolo/inspection/task/${taskId}`);
};

type Options = {
  mutationConfig?: MutationConfig<typeof discardInspection>;
};

export const useDiscardInspection = ({ mutationConfig }: Options = {}) => {
  const { onSuccess, ...rest } = mutationConfig || {};

  return useMutation({
    mutationFn: discardInspection,
    onSuccess: (...args) => {
      const [, variables] = args;

      queryClient.removeQueries({
        queryKey: getTaskQueryOptions(variables.taskId).queryKey,
      });
      onSuccess?.(...args);
    },
    ...rest,
  });
};
