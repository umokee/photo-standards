import { queryKeys } from "@/lib/query-keys";
import { SystemStatsResponse } from "@/types/contracts/system";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

export function useSystemStatsLive() {
  const qc = useQueryClient();

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/system/stats`);

    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data) as SystemStatsResponse;
      qc.setQueryData(queryKeys.settings.system_stats(), data);
    };

    return () => ws.close();
  }, [qc]);
}
