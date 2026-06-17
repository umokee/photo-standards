
import { FileImage, Upload } from "lucide-react";
import { ChangeEvent, DragEvent, useRef, useState } from "react";
import s from "./inspection-dropzone.module.scss";

type Props = {
  disabled: boolean;
  onFileSelect: (file: File) => void;
};

export function InspectionDropzone({ disabled, onFileSelect }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onFileSelect(file);
    event.target.value = "";
  };

  const handleDragOver = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    const file = Array.from(event.dataTransfer.files).find((item) => item.type.startsWith("image/"));
    if (file) onFileSelect(file);
  };

  const openPicker = () => {
    if (!disabled) inputRef.current?.click();
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        disabled={disabled}
        onChange={handleChange}
      />
      <button
        type="button"
        className={s.root}
        data-dragging={isDragging ? "true" : "false"}
        disabled={disabled}
        onClick={openPicker}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <span className={s.icon}>
          <Upload />
        </span>
        <span className={s.body}>
          <strong>Загрузить изображение</strong>
          <small>Перетащи файл сюда или выбери изображение с диска</small>
        </span>
        <span className={s.hint}><FileImage /> JPG · PNG · WEBP</span>
      </button>
    </>
  );
}
