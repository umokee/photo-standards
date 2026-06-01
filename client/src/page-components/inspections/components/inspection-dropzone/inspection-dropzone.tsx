import { Upload } from "lucide-react";
import { ChangeEvent, useRef } from "react";
import s from "./inspection-dropzone.module.scss";

type Props = {
  disabled: boolean;
  onFileSelect: (file: File) => void;
};

export function InspectionDropzone({ disabled, onFileSelect }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onFileSelect(file);
    event.target.value = "";
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
      <button type="button" className={s.root} disabled={disabled} onClick={openPicker}>
        <div className={s.icon}>
          <Upload size={28} />
        </div>
        <span className={s.title}>Загрузить изображение</span>
      </button>
    </>
  );
}
