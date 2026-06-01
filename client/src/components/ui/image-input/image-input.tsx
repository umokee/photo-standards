import clsx from "clsx";
import { Upload, X } from "lucide-react";
import { type MouseEvent, useRef, useState } from "react";
import s from "./image-input.module.scss";

const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024;
const SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png"];

interface Props {
  label?: string;
  error?: string;
  multiple?: boolean;
  value: File[] | null;
  onChange: (value: File[] | null) => void;
}

export default function ImageInput({
  label,
  error,
  multiple = false,
  value,
  onChange,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const resolvedError = localError ?? error ?? null;

  const handleFiles = (files: File[]) => {
    const nextFiles = multiple ? files : files.slice(0, 1);

    if (nextFiles.length === 0) {
      setLocalError(null);
      onChange(null);
      return;
    }

    for (const file of nextFiles) {
      if (!SUPPORTED_IMAGE_TYPES.includes(file.type)) {
        setLocalError(`${file.name} - только JPG и PNG файлы`);
        return;
      }

      if (file.size > MAX_FILE_SIZE_BYTES) {
        setLocalError(`${file.name} - слишком большой файл`);
        return;
      }
    }

    setLocalError(null);
    onChange(nextFiles);
  };

  const handleClear = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();

    setLocalError(null);
    onChange(null);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  return (
    <div className={s.root}>
      {label && <label className={s.label}>{label}</label>}

      <input
        ref={inputRef}
        hidden
        type="file"
        multiple={multiple}
        accept="image/jpeg,image/png"
        onChange={(event) => handleFiles(Array.from(event.target.files ?? []))}
      />

      <div
        className={clsx(s.zone, {
          [s.zoneDragging]: isDragging,
          [s.zoneError]: !isDragging && !!resolvedError,
        })}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          handleFiles(Array.from(event.dataTransfer.files));
        }}
      >
        {resolvedError ? (
          <span className={s.errorText}>{resolvedError}</span>
        ) : value && value.length > 0 ? (
          <div className={s.files}>
            <span className={s.filesCount}>Выбрано: {value.length} фото</span>

            <button
              type="button"
              className={s.clearButton}
              aria-label="Очистить выбранные изображения"
              onClick={handleClear}
            >
              <X size={16} />
            </button>
          </div>
        ) : (
          <>
            <Upload className={s.icon} size={28} />
            <span className={s.title}>Перетащите или выберите изображения</span>
            <span className={s.subtitle}>JPG, PNG · максимум 20 Мб</span>
          </>
        )}
      </div>
    </div>
  );
}
