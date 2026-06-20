# UI polish v119 groups CSS prune report

Generated: `2026-06-21 03:27:48`
Project root: `/home/user/jkljlk/photo-standards-db`

## Changed files

- `client/src/app/routes/groups/_project-assets-strict.module.scss`

## Importers checked

- `client/src/app/routes/groups/_assets-overview.tsx`
- `client/src/app/routes/groups/_group-detail.tsx`
- `client/src/app/routes/groups/_groups-index.tsx`
- `client/src/app/routes/groups/_standard-detail.tsx`

## What changed

- Removed unused CSS-module selectors from `_project-assets-strict.module.scss`.
- Checked all active TSX importers of the shared groups SCSS module, not only `_groups-index.tsx`.
- Kept every class referenced by active routes: groups list, assets overview, standard detail and group shell.
- Did not change TSX, routes, API hooks, React Query or component contracts.

## Metrics

- `importers`: `4`
- `used_classes_across_importers`: `107`
- `scss_classes_after_prune`: `107`
- `removed_unique_classes`: `11`
- `scss_bytes`: `31865`

## Removed stale selectors

- `.avatar` × 1
- `.metaGrid` × 4
- `.projectCard` × 3
- `.projectCardActions` × 1
- `.projectCardDeleteAction` × 1
- `.projectCardLink` × 1
- `.projectCardOpenAction` × 1
- `.projectGrid` × 1
- `.projectTitle` × 3
- `.projectTop` × 1
- `.summaryGridSix` × 3

## Manual check

- Groups list should keep the v118 layout.
- Assets overview should keep metric cards and task rows.
- Standard detail should keep photo cards, pool rail, header actions and status strips.
- Group detail route should keep shell spacing.
