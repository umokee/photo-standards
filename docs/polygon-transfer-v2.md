# Polygon Transfer V2: registration-first pipeline

## Цель

Новый пайп нужен не для того, чтобы добавить ещё один fallback в старый matcher. Его цель — проверить более чистую архитектуру переноса полигонов без YOLO:

1. сначала зарегистрировать всю сцену `reference -> target`;
2. перенести эталонные полигоны как seed-зоны;
3. локально уточнить каждый слот только по окружению;
4. принять полигон только при подтверждённой локальной регистрации;
5. иначе вернуть `unconfirmed`, а не угадывать.

## Почему старый пайп застрял

Старый matcher смешивает несколько разных задач:

- глобальная регистрация сцены;
- перенос expected slot;
- локальный rescue;
- shadow/debug кандидаты;
- hidden release policy;
- anchor/Yolo-oriented диагностика;
- candidate agreement между разными эвристиками.

Из-за этого точка отказа неочевидна. Например `none` может означать не отсутствие точек, а то, что seed-полигон не построился; `unsafe_hidden` может означать и реальный риск, и то, что один из fallback-кандидатов не прошёл policy.

V2 должен отвечать на один вопрос: можно ли надёжно перенести координаты эталона в координаты фото по окружающему контексту.

## Принцип V2

### 1. Scene registration

Входом являются реальные SuperPoint/LightGlue пары из `LocalProjectionData`.

V2 сначала пытается получить scene transform:

- если уже есть надёжная homography, использует её;
- иначе пробует `estimateAffinePartial2D` по глобальным матчам;
- если scene consensus слабый, все объекты получают `v2_unconfirmed`.

Scene transform не является финальным ответом по объекту. Он нужен только как система координат.

### 2. Seed projection

Каждый reference polygon переносится через scene transform:

```text
seed_polygon = scene_transform(reference_polygon)
```

Seed polygon — это стартовая зона, не подтверждённый missing polygon.

### 3. Context-only local registration

Reference image сначала warps в координаты target. После этого для каждого seed:

1. берётся crop вокруг seed;
2. все expected-полигоны внутри crop заливаются median/background цветом;
3. строится ring mask, где object interior исключён;
4. оценивается локальный сдвиг через phase correlation;
5. затем пробуется ECC translation refinement с mask;
6. результат принимается только если локальный score и geometry gates нормальные.

Главная идея: локальный refine должен смотреть не на сам объект, а на окружение вокруг его места.

### 4. Acceptance policy

V2 принимает только:

```text
scene transform ok
+ seed polygon ok
+ context-only local registration ok
+ refined polygon visible
+ shift not too large
+ no excessive overlap with neighbor slots
```

V2 не принимает как финал:

- raw `expected_slot`;
- hidden shadow;
- full crop matching по объекту;
- candidate agreement без transform quality;
- локальный сдвиг без ring/context score.

## Что смотреть в отчёте

При запуске теста с флагом:

```bash
--projection-pipeline v2
```

в debug/results должны появиться поля:

```text
v2_pipeline
v2_scene_registration_accepted
v2_scene_registration_mode
v2_scene_raw_matches
v2_scene_inliers
v2_scene_inlier_ratio
v2_scene_median_error
v2_local_registration_attempted
v2_local_registration_accepted
v2_local_registration_source
v2_ring_fraction
v2_phase_response
v2_ecc_score
v2_shift_factor
v2_max_other_overlap
```

Сначала нужно сравнить не только accuracy, но и распределение причин:

- стало ли меньше `none`;
- стало ли меньше `unsafe_hidden`;
- сколько объектов стало `v2_context_ecc` / `v2_context_phase`;
- сколько объектов уходит в `v2_scene_registration_failed`;
- сколько объектов уходит в `v2_local_context_registration_failed`;
- вырос ли `dangerous`.

## Ожидаемый результат эксперимента

V2 может сначала дать меньший coverage, чем старый matcher. Это нормально. Его задача — показать чистый baseline без старых эвристик.

Хороший знак:

```text
v2_context_ecc / v2_context_phase имеют высокий pass-rate и низкий dangerous
```

Плохой знак:

```text
много v2_scene_registration_failed
```

Тогда проблема в глобальной регистрации сцены.

Другой плохой знак:

```text
scene registration ok, но много v2_local_context_registration_failed
```

Тогда проблема в локальном контексте: либо crop/seed не туда попадает, либо вокруг объекта нет уникального окружения.

## Дальше

Если V2 подтвердит идею, следующий шаг — заменить phase/ECC блок на несколько кандидатов local registration:

1. phase translation;
2. ECC translation;
3. ECC euclidean;
4. masked local LightGlue только как validator;
5. при необходимости LoFTR/детектор-free matcher как второй источник матчей.

Но эти методы должны оставаться внутри одного registration-first пайпа, а не превращаться в независимые fallback-и.

## V81: safety-first hardening

Первый прогон V2 показал правильную форму пайпа: `v2_context_ecc` давал высокий pass-rate, но часть локальных ECC-переносов уезжала на похожий соседний слот. Поэтому V81 делает V2 более строгим.

Изменения V81:

- scene homography больше не принимается по условию `inliers OR ratio`; теперь нужны одновременно достаточные `raw matches`, `inliers`, `inlier_ratio` и низкая `median_error`;
- affine fallback использует те же scene-quality правила;
- ECC/phase refine должен быть ближе к seed: снижены лимиты `shift_factor`, `center_factor` и `max_other_overlap`;
- большой локальный сдвиг принимается только при сильном ECC score;
- phase-only результат почти всегда diagnostic: он может стать финальным только при сильном phase response и малом сдвиге;
- добавлена диагностика `v2_homography_*`, `v2_phase_ecc_delta_factor`, `v2_center_factor`.

Цель V81 — сначала убрать dangerous-переносы. Если coverage временно падает, это нормально: следующий шаг должен поднимать coverage через более сильный scene registration, а не через угадывающие fallback-и.

## V82: calibrated safety gates

V81 showed that the safety gates work, but were too strict: dangerous transfers
dropped strongly while many good single and multi cases became unconfirmed. V82
keeps the same registration-first architecture and calibrates thresholds instead
of adding new fallback branches. Scene registration can pass either by inlier
ratio or by strong absolute consensus with tight median error; local ECC remains
bounded by shift, center movement, phase/ECC agreement, visibility, and neighbor
overlap.
