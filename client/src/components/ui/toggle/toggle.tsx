import s from "./toggle.module.scss";

interface Props {
  label?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export default function Toggle({ label, checked, onChange }: Props) {
  return (
    <label className={s.root}>
      <input
        className={s.input}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className={s.track}>
        <span className={s.thumb} />
      </span>
      {label && <span className={s.label}>{label}</span>}
    </label>
  );
}
