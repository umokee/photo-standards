import clsx from "clsx";
import s from "./color-picker.module.scss";

const PRESET_HUES = [0, 18, 36, 52, 90, 132, 168, 210, 248, 282, 318];

interface Props {
  hue: number;
  onChange: (hue: number) => void;
}

export const ColorPicker = ({ hue, onChange }: Props) => {
  return (
    <div className={s.root}>
      <div className={s.presets}>
        {PRESET_HUES.map((presetHue) => {
          const isActive = Math.abs(hue - presetHue) < 8;

          return (
            <button
              key={presetHue}
              type="button"
              className={clsx(s.preset, isActive && s.presetActive)}
              style={{ background: `hsl(${presetHue}, 65%, 55%)` }}
              onClick={() => onChange(presetHue)}
            />
          );
        })}
      </div>

      <label className={s.sliderBlock}>
        <span className={s.sliderLabel}>Оттенок: {Math.round(hue)}&deg;</span>
        <input
          type="range"
          min={0}
          max={360}
          value={hue}
          className={s.slider}
          aria-label="Оттенок цвета"
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </label>
    </div>
  );
};
