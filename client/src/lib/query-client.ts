import { MutationCache, QueryCache, QueryClient } from "@tanstack/react-query";
import { handleAppError, queryConfig } from "./react-query";

export const queryClient = new QueryClient({
  defaultOptions: queryConfig,
  queryCache: new QueryCache({
    onError: handleAppError,
  }),
  mutationCache: new MutationCache({
    onError: handleAppError,
  }),
});
