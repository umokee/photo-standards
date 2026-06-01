import { Image, Layer } from "react-konva";

type Props = {
  image: CanvasImageSource | undefined;
  x: number;
  y: number;
  width: number;
  height: number;
};

export function CanvasImage({ image, x, y, width, height }: Props) {
  return (
    <Layer>
      <Image image={image} x={x} y={y} width={width} height={height} />
    </Layer>
  );
}
