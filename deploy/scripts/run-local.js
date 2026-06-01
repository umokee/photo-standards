#!/usr/bin/env node

const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const readline = require("node:readline");

const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const SERVER_DIR = path.join(PROJECT_ROOT, "server");
const CLIENT_DIR = path.join(PROJECT_ROOT, "client");

const isWindows = process.platform === "win32";
const pythonBin = isWindows
  ? path.join(SERVER_DIR, ".venv", "Scripts", "python.exe")
  : path.join(SERVER_DIR, ".venv", "bin", "python");
const npmBin = isWindows ? "npm.cmd" : "npm";

const serverHost = process.env.SERVER_HOST || "0.0.0.0";
const serverPort = process.env.SERVER_PORT || "3001";
const clientHost = process.env.CLIENT_HOST || "0.0.0.0";
const clientPort = process.env.CLIENT_PORT || "5173";
const serverBaseUrl = process.env.SERVER_BASE_URL || `http://127.0.0.1:${serverPort}`;
const serverHealthUrl = process.env.SERVER_HEALTH_URL || `${serverBaseUrl}/api/health`;
const serverReload = process.env.SERVER_RELOAD !== "false";
const readyTimeoutMs = Number(process.env.SERVER_READY_TIMEOUT_MS || 120000);
const shutdownTimeoutMs = Number(process.env.LAUNCHER_SHUTDOWN_TIMEOUT_MS || 5000);
const runInitDb =
  process.env.RUN_INIT_DB === "1" ||
  process.env.RUN_INIT_DB === "true" ||
  process.env.RUN_INIT_DB === "yes";
const disableColor = process.env.NO_COLOR === "1" || process.env.NO_COLOR === "true";

const ANSI = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
  gray: "\x1b[90m",
};

const SERVICE_COLORS = {
  launcher: ANSI.blue,
  server: ANSI.cyan,
  client: ANSI.green,
  "init-db": ANSI.magenta,
};

const children = [];
let shuttingDown = false;

function supportsColor(target) {
  return Boolean(target?.isTTY) && !disableColor;
}

function paint(target, text, ...codes) {
  if (!text || !supportsColor(target) || codes.length === 0) {
    return text;
  }

  return `${codes.join("")}${text}${ANSI.reset}`;
}

function serviceTag(target, service) {
  const color = SERVICE_COLORS[service] || ANSI.blue;
  return paint(target, `[${service}]`, ANSI.bold, color);
}

function formatTimestamp(date = new Date()) {
  const pad = (value) => String(value).padStart(2, "0");

  return [
    `${pad(date.getDate())}.${pad(date.getMonth() + 1)}.${date.getFullYear()}`,
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`,
  ].join(" ");
}

function logLine(target, service, message) {
  const timestamp = paint(target, formatTimestamp(), ANSI.dim);
  target.write(`${serviceTag(target, service)} ${timestamp} ${message}\n`);
}

function failFast(message, code = 1) {
  logLine(process.stderr, "launcher", message);
  process.exit(code);
}

function ensureExists(targetPath, label) {
  if (!fs.existsSync(targetPath)) {
    failFast(`Не найден ${label}: ${targetPath}`);
  }
}

function ensureProjectLayout() {
  ensureExists(SERVER_DIR, "каталог server");
  ensureExists(CLIENT_DIR, "каталог client");
  ensureExists(path.join(CLIENT_DIR, "package.json"), "client/package.json");
  ensureExists(path.join(SERVER_DIR, "src", "main.py"), "server/src/main.py");
  ensureExists(pythonBin, "Python из server/.venv");

  if (runInitDb) {
    ensureExists(path.join(SERVER_DIR, "src", "init_db.py"), "server/src/init_db.py");
  }
}

function buildBaseEnv() {
  return {
    ...process.env,
    DEBUG: process.env.DEBUG || "true",
    SERVER_RELOAD: serverReload ? "true" : "false",
  };
}

function assertPortFree(host, port, label) {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();

    probe.once("error", (error) => {
      reject(
        new Error(
          `${label}: порт ${host}:${port} недоступен (${error.code || error.message})`
        )
      );
    });

    probe.listen(Number(port), host, () => {
      probe.close(() => resolve());
    });
  });
}

function prefixStream(name, stream, target) {
  if (!stream) return;

  const rl = readline.createInterface({ input: stream });
  rl.on("line", (line) => {
    if (!line.trim()) return;
    logLine(target, name, line);
  });
}

function runStep(name, command, args, options = {}) {
  logLine(process.stdout, name, `$ ${command} ${args.join(" ")}`);

  const result = spawnSync(command, args, {
    cwd: options.cwd || PROJECT_ROOT,
    env: options.env || process.env,
    stdio: ["ignore", "pipe", "pipe"],
    encoding: "utf8",
    shell: isWindows,
  });

  for (const line of (result.stdout || "").split(/\r?\n/)) {
    if (line.trim()) {
      logLine(process.stdout, name, line);
    }
  }

  for (const line of (result.stderr || "").split(/\r?\n/)) {
    if (line.trim()) {
      logLine(process.stderr, name, line);
    }
  }

  if (result.error) {
    failFast(`${name}: ${result.error.message}`);
  }

  if (result.status !== 0) {
    failFast(`${name}: команда завершилась с кодом ${result.status ?? 1}`);
  }
}

function spawnService(name, command, args, options = {}) {
  logLine(process.stdout, name, `$ ${command} ${args.join(" ")}`);

  const child = spawn(command, args, {
    cwd: options.cwd || PROJECT_ROOT,
    env: options.env || process.env,
    stdio: ["ignore", "pipe", "pipe"],
    detached: !isWindows,
    shell: isWindows,
  });

  child.on("error", (error) => {
    if (!shuttingDown) {
      void fail(`${name}: ${error.message}`);
    }
  });

  prefixStream(name, child.stdout, process.stdout);
  prefixStream(name, child.stderr, process.stderr);

  child.on("exit", (code, signal) => {
    if (!shuttingDown) {
      const reason = signal ? `signal ${signal}` : `code ${code ?? 0}`;
      void fail(`${name} завершился раньше времени (${reason})`);
    }
  });

  children.push({ name, child });
  return child;
}

function signalChild(entry, signal) {
  const { child } = entry;

  if (!child || child.exitCode !== null) {
    return;
  }

  if (isWindows) {
    spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], {
      stdio: "ignore",
      shell: false,
    });
    return;
  }

  if (typeof child.pid === "number") {
    try {
      process.kill(-child.pid, signal);
      return;
    } catch {}
  }

  try {
    child.kill(signal);
  } catch {}
}

function waitForExit(child, timeoutMs) {
  if (!child || child.exitCode !== null) {
    return Promise.resolve(true);
  }

  return new Promise((resolve) => {
    const onExit = () => {
      clearTimeout(timer);
      resolve(true);
    };

    const timer = setTimeout(() => {
      child.off("exit", onExit);
      resolve(false);
    }, timeoutMs);

    child.once("exit", onExit);
  });
}

async function shutdown(code = 0) {
  if (shuttingDown) {
    return;
  }

  shuttingDown = true;

  for (const entry of children) {
    signalChild(entry, "SIGTERM");
  }

  await Promise.all(
    children.map(async ({ child, name }) => {
      const exited = await waitForExit(child, shutdownTimeoutMs);
      if (exited) {
        return;
      }

      logLine(process.stderr, "launcher", `${name} не завершился по SIGTERM, отправляем SIGKILL`);
      signalChild({ child }, "SIGKILL");
      await waitForExit(child, 2000);
    })
  );

  process.exit(code);
}

async function fail(message, code = 1) {
  logLine(process.stderr, "launcher", message);
  await shutdown(code);
}

async function waitForServerReady() {
  const startedAt = Date.now();

  while (!shuttingDown) {
    if (Date.now() - startedAt > readyTimeoutMs) {
      await fail(`server не поднялся за ${Math.round(readyTimeoutMs / 1000)}с`);
      return;
    }

    try {
      const response = await fetch(serverHealthUrl);
      if (response.ok) {
        logLine(process.stdout, "launcher", `server готов: ${serverHealthUrl}`);
        return;
      }
    } catch {}

    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

async function main() {
  ensureProjectLayout();

  process.on("SIGINT", () => {
    void shutdown(0);
  });

  process.on("SIGTERM", () => {
    void shutdown(0);
  });

  const baseEnv = buildBaseEnv();

  await assertPortFree(serverHost, serverPort, "backend");
  await assertPortFree(clientHost, clientPort, "frontend");

  if (runInitDb) {
    runStep("init-db", pythonBin, ["src/init_db.py"], {
      cwd: SERVER_DIR,
      env: baseEnv,
    });
  }

  spawnService("server", pythonBin, ["src/main.py"], {
    cwd: SERVER_DIR,
    env: {
      ...baseEnv,
      SERVER_HOST: serverHost,
      SERVER_PORT: serverPort,
    },
  });

  logLine(process.stdout, "launcher", `ждём backend healthcheck: ${serverHealthUrl}`);
  await waitForServerReady();

  spawnService(
    "client",
    npmBin,
    ["run", "dev", "--", "--host", clientHost, "--port", clientPort, "--strictPort"],
    {
      cwd: CLIENT_DIR,
      env: process.env,
    }
  );

  logLine(
    process.stdout,
    "launcher",
    `готово: frontend=http://127.0.0.1:${clientPort} backend=${serverBaseUrl}`
  );

  if (!runInitDb) {
    logLine(
      process.stdout,
      "launcher",
      "БД не инициализировалась автоматически. Используй npm run db:bootstrap или npm run db:migrate."
    );
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);

  if (children.length > 0) {
    void fail(message, 1);
    return;
  }

  failFast(message, 1);
});
