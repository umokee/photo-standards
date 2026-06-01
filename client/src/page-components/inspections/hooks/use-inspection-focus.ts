import type { InspectionRealtimeStatus, InspectionTaskResult } from "@/types/contracts";
import { useEffect, useState } from "react";
import type { InspectionFocus } from "../lib/inspection-context";

type InspectionResultLike = InspectionTaskResult | InspectionRealtimeStatus | null;

export function useInspectionFocus(result: InspectionResultLike): InspectionFocus {
  const [activeMatchKey, setActiveMatchKey] = useState<string | null>(null);

  useEffect(() => {
    if (!result?.details?.length) {
      setActiveMatchKey(null);
      return;
    }

    setActiveMatchKey((current) => {
      if (current !== null) {
        const idx = Number.parseInt(current, 10);

        if (!Number.isNaN(idx) && idx >= 0 && idx < result.details.length) {
          return current;
        }
      }

      const firstIssueIdx = result.details.findIndex((detail) => detail.status !== "ok");

      if (firstIssueIdx >= 0) {
        return String(firstIssueIdx);
      }

      return result.details.length > 0 ? "0" : null;
    });
  }, [result]);

  return {
    activeMatchKey,
    setActiveMatchKey,
  };
}
