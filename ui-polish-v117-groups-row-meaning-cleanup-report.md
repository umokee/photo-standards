# UI polish v117 groups row meaning cleanup report

Generated: `2026-06-21 03:16:14`
Project root: `/home/user/jkljlk/photo-standards-db`

## Changed files

- `client/src/app/routes/groups/_groups-index.tsx`
- `client/src/app/routes/groups/_project-assets-strict.module.scss`

## What changed

- Removed the first-letter avatar from every group row.
- Removed the separate next-step/history block from the row.
- Removed `Посмотреть историю`/`Открыть историю` wording from the groups list.
- Replaced the cramped shared-looking `можно проверять + 5/5` badge with a local setup block.
- Reworded the ready state to `Проверка доступна` and the incomplete state to `Нужно настроить`.
- Kept row click as the only overview navigation.
- Kept only real management actions on the right: edit and delete.

## Metrics

- `groups_index_used_classes`: `41`
- `groups_scss_classes`: `119`
- `v117_classes_used`: `26`
- `removed_avatar`: `1`
- `removed_next_step_block`: `1`
- `removed_duplicate_nav_actions`: `2`
- `groups_scss_bytes`: `34237`

## Manual check

- No avatar/letter at the left of the row.
- No `Обзор`, `Продолжить`, `Посмотреть историю`, or `Открыть историю` buttons/text in a row.
- Setup block reads as local row content, not as a generic badge.
- `Проверка доступна` and `Настройка 5/5` should not visually collide.
- Stats and actions should remain aligned on desktop and stacked on mobile.
