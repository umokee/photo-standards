import { ImageDown } from "lucide-react";
import { useState } from "react";

interface Props {
  src?: string | null;
  className?: string;
  iconSize?: number;
}

export default function ImageWithFallback({ src, className, iconSize = 24 }: Props) {
  const [error, setError] = useState(false);
  if (!src || error) {
    return (
      <div className={className}>
        <ImageDown size={iconSize} />
      </div>
    );
  }
  return (
    <img
      className={className}
      src={src}
      loading="lazy"
      decoding="async"
      onError={() => setError(true)}
    />
  );
}
