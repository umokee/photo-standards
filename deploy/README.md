# Deploy

This directory contains deployment helpers for PhotoStandards.

## What is inside

- `env/server.env.example` - server `.env` template.
- `nginx/enterprise.conf` - nginx server block for an enterprise server.
- `nginx/local-test.conf.template` - template used by local production-test scripts.
- `systemd/photo-standards-backend.service` - backend service example.
- `scripts/run-local.js` - local dev launcher for backend and frontend.
- `scripts/deploy.mjs` - cross-platform deploy runner used by npm scripts.

## Local development

From the project root:

```bash
npm run dev
```

The same launcher is available through deploy commands:

```bash
npm run deploy:dev
```

Both commands use `deploy/scripts/run-local.js`.

## Quick local production test

From the project root:

```bash
npm run deploy:prod:local
```

Keep this command running. In another terminal:

```bash
npm run deploy:prod:check
npm run deploy:prod:stop
```

Defaults:

- frontend through nginx: `http://127.0.0.1:8088`
- backend: `http://127.0.0.1:3011`
- generated files and logs: `.deploy/runtime/`

Useful overrides:

```bash
FRONTEND_PORT=8090 BACKEND_PORT=3012 npm run deploy:prod:local
SKIP_BUILD=1 RUN_MIGRATIONS=0 npm run deploy:prod:local
```

## Enterprise install outline

1. Copy the project to `/opt/photo-standards-db`.
2. Copy `deploy/env/server.env.example` to `server/.env` and edit values.
3. Install backend and frontend dependencies.
4. Run `npm run deploy:init-db`.
5. Build frontend with `npm --prefix client run build`.
6. Copy `client/dist/` to `/var/www/photo-standards/`.
7. Install `deploy/systemd/photo-standards-backend.service` to `/etc/systemd/system/` and edit paths if needed.
8. Install `deploy/nginx/enterprise.conf` to nginx config and edit `server_name`.
9. Start/reload services.
