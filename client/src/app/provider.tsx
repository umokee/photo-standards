import { Notifications } from "@/components/ui/notifications/notifications";
import { SidebarProvider } from "@/context/sidebar-context";
import { queryClient } from "@/lib/query-client";
import { QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "react-router-dom";
import { router } from "./router";

export default function AppProvider() {
  return (
    <>
      <QueryClientProvider client={queryClient}>
        <SidebarProvider>
          <RouterProvider router={router} />
        </SidebarProvider>
        <Notifications />
      </QueryClientProvider>
    </>
  );
}
