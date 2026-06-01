import clsx from "clsx";
import { Field } from "../field/field";
import s from "./input.module.scss";

interface Props {
  label?: string;
  placeholder?: string;
  value: string;
  error?: string;
  noMargin?: boolean;
  type?: "text" | "password" | "number";
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: string) => void;
}

export default function Input({
  label,
  placeholder,
  value,
  error,
  type = "text",
  noMargin,
  min,
  max,
  step,
  onChange,
}: Props) {
  const handleChange = (nextValue: string) => {
    if (type !== "number") {
      onChange(nextValue);
      return;
    }

    if (nextValue === "") {
      onChange(nextValue);
      return;
    }

    const nextNumber = Number(nextValue);
    if (Number.isNaN(nextNumber)) return;

    if (max !== undefined && nextNumber > max) {
      onChange(String(max));
      return;
    }

    onChange(nextValue);
  };

  const handleBlur = () => {
    if (type !== "number" || value === "") return;

    const nextNumber = Number(value);
    if (Number.isNaN(nextNumber)) return;

    if (min !== undefined && nextNumber < min) {
      onChange(String(min));
      return;
    }

    if (max !== undefined && nextNumber > max) {
      onChange(String(max));
    }
  };

  const handleKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (event) => {
    if (type !== "number") return;

    if (["e", "E", "+", "-"].includes(event.key)) {
      event.preventDefault();
    }
  };

  return (
    <Field label={label} error={error} noMargin={noMargin}>
      <input
        className={clsx(s.root, error && s.fieldError)}
        type={type}
        placeholder={placeholder}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => handleChange(e.target.value)}
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
      />
    </Field>
  );
}
