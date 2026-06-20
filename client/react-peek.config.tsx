import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      staleTime: Infinity,
    },
  },
});

export default {
  wrapper: ({ children }: { children: ReactNode }) => {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  },

  aliases: {
    "@": "./src",
    "@components": "./src/components",
    "@ui": "./src/components/ui",
    "@lib": "./src/lib",
    "@styles": "./src/styles",
  },

  css: {
    scssLoadPaths: ["src"],
  },

  globalCss: [],
};
