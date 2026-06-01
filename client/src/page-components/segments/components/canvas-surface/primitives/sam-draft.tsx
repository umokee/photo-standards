import { Circle, Layer, Line } from "react-konva";

type Props = {
  positiveCanvas: number[][];
  negativeCanvas: number[][];
  draftPolygonCanvas: number[][] | null;
  stroke: string;
  viewportScale?: number;
  isLoading: boolean;
};

export function SamDraft({
  positiveCanvas,
  negativeCanvas,
  draftPolygonCanvas,
  stroke,
  viewportScale = 1,
  isLoading,
}: Props) {
  const visualScale = 1 / Math.max(viewportScale, 1);
  const pointRadius = 5 * visualScale;
  const polygonWidth = 2 * visualScale;
  const polygonOutlineWidth = 3.5 * visualScale;

  return (
    <Layer listening={false}>
      {draftPolygonCanvas && draftPolygonCanvas.length >= 3 && (
        <>
          <Line
            points={draftPolygonCanvas.flat()}
            closed
            stroke="rgba(255, 255, 255, 0.9)"
            strokeWidth={polygonOutlineWidth}
            lineJoin="round"
            lineCap="round"
          />
          <Line
            points={draftPolygonCanvas.flat()}
            closed
            stroke={stroke}
            fill={stroke}
            opacity={isLoading ? 0.35 : 0.18}
            strokeWidth={polygonWidth}
            lineJoin="round"
            lineCap="round"
          />
        </>
      )}
      {positiveCanvas.map(([x, y], i) => (
        <Circle
          key={`pos-${i}`}
          x={x}
          y={y}
          radius={pointRadius}
          fill="#22c55e"
          stroke="white"
          strokeWidth={1.5 * visualScale}
        />
      ))}
      {negativeCanvas.map(([x, y], i) => (
        <Circle
          key={`neg-${i}`}
          x={x}
          y={y}
          radius={pointRadius}
          fill="#ef4444"
          stroke="white"
          strokeWidth={1.5 * visualScale}
        />
      ))}
    </Layer>
  );
}
