import { clamp } from "@/page-components/segments/lib/canvas";
import { useEffect, useRef, useState } from "react";

interface ImageRect {
  scale: number;
  width: number;
  height: number;
  offsetX: number;
  offsetY: number;
}

export default function useImageLayout(image: HTMLImageElement | null) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([e]) =>
      setSize({ width: e.contentRect.width, height: e.contentRect.height })
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const imageRect: ImageRect = image
    ? (() => {
        const scale = Math.min(
          (size.width - 32) / image.naturalWidth,
          (size.height - 32) / image.naturalHeight
        );
        const width = image.naturalWidth * scale;
        const height = image.naturalHeight * scale;
        return {
          scale,
          width,
          height,
          offsetX: (size.width - width) / 2,
          offsetY: (size.height - height) / 2,
        };
      })()
    : { scale: 1, width: 0, height: 0, offsetX: 0, offsetY: 0 };

  const toImage = (cx: number, cy: number): [number, number] => [
    Math.round((cx - imageRect.offsetX) / imageRect.scale),
    Math.round((cy - imageRect.offsetY) / imageRect.scale),
  ];

  const toCanvas = (ix: number, iy: number): [number, number] => [
    imageRect.offsetX + ix * imageRect.scale,
    imageRect.offsetY + iy * imageRect.scale,
  ];

  const clampToImage = (cx: number, cy: number): [number, number] => [
    clamp(cx, imageRect.offsetX, imageRect.offsetX + imageRect.width),
    clamp(cy, imageRect.offsetY, imageRect.offsetY + imageRect.height),
  ];

  return { containerRef, size, imageRect, toImage, toCanvas, clampToImage };
}
