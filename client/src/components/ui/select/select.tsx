import clsx from "clsx";
import { Field } from "../field/field";
import s from "./select.module.scss";

interface Option {
  value: string;
  label: string;
}

interface Props {
  label?: string;
  options: Option[];
  value: string | null;
  placeholder?: string;
  error?: string;
  disabled?: boolean;
  noMargin?: boolean;
  onChange: (value: string) => void;
}

export default function Select({
  label,
  options,
  value,
  placeholder,
  error,
  disabled,
  noMargin,
  onChange,
}: Props) {
  return (
    <Field label={label} error={error} noMargin={noMargin}>
      <select
        className={clsx(s.root, error && s.fieldError)}
        disabled={disabled}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
      >
        {placeholder && <option value="">{placeholder}</option>}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </Field>
  );
}
