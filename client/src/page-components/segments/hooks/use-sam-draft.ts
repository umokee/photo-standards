import { useSamPredict, useSamWarmup } from "@/page-components/segments/api/sam-predict";
import { useCallback, useEffect, useRef, useState } from "react";

export function useSamDraft({ isActive, imageId }: { isActive: boolean; imageId: string }) {
  const [positive, setPositive] = useState<[number, number][]>([]);
  const [negative, setNegative] = useState<[number, number][]>([]);
  const [draftPolygon, setDraftPolygon] = useState<number[][] | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const predict = useSamPredict();
  const debounceRef = useRef<number | null>(null);
  const genRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const warmup = useSamWarmup();

  const reset = useCallback(() => {
    setPositive([]);
    setNegative([]);
    setDraftPolygon(null);
    setIsLoading(false);
    if (debounceRef.current !== null) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    genRef.current++;
  }, []);

  useEffect(() => {
    if (!isActive) reset();
  }, [isActive, reset]);

  useEffect(() => {
    if (!isActive || !imageId) return;
    warmup.mutate({ imageId });
  }, [isActive, imageId]);

  useEffect(() => {
    if (!isActive) return;
    if (positive.length === 0) {
      setDraftPolygon(null);
      return;
    }

    if (debounceRef.current !== null) clearTimeout(debounceRef.current);

    debounceRef.current = window.setTimeout(() => {
      if (abortRef.current) abortRef.current.abort();
      abortRef.current = new AbortController();

      const gen = ++genRef.current;
      setIsLoading(true);
      predict.mutate(
        { imageId, positive, negative },
        {
          onSettled: (data) => {
            if (gen !== genRef.current) return;
            setIsLoading(false);
            setDraftPolygon(data?.points ?? null);
          },
        }
      );
    }, 150);

    return () => {
      if (debounceRef.current !== null) clearTimeout(debounceRef.current);
    };
  }, [isActive, imageId, positive, negative]);

  const addPositive = useCallback((p: [number, number]) => {
    setPositive((prev) => [...prev, p]);
  }, []);

  const addNegative = useCallback((p: [number, number]) => {
    setNegative((prev) => [...prev, p]);
  }, []);

  const removeLast = useCallback(() => {
    if (negative.length > 0) {
      setNegative((prev) => prev.slice(0, -1));
      return;
    }
    setPositive((prev) => prev.slice(0, -1));
  }, [negative.length]);

  return {
    positive,
    negative,
    draftPolygon,
    isLoading,
    addPositive,
    addNegative,
    removeLast,
    reset,
  };
}
