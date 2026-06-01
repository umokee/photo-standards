#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const scriptDir = path.dirname(__filename);
const deployDir = path.resolve(scriptDir, "..");
const projectRoot = path.resolve(deployDir, "..");
const runtimeDir = path.join(projectRoot, ".deploy", "runtime");

const isWindows = process.platform === "win32";
const npmBin = isWindows ? "npm.cmd" : "npm";
const pythonBin = isWindows
  ? path.join(projectRoot, "server", ".venv", "Scripts", "python.exe")
  : path.join(projectRoot, "server", ".venv", "bin", "python");

const backendHost = process.env.BACKEND_HOST || "127.0.0.1";
const backendPort = process.env.BACKEND_PORT || "3011";
const frontendHost = process.env.FRONTEND_HOST || "127.0.0.1";
const frontendPort = process.env.FRONTEND_PORT || "8088";
const backendUrl = `http://${backendHost}:${backendPort}`;
const frontendUrl = `http://${frontendHost}:${frontendPort}`;

let backendProcess = null;
let nginxProcess = null;
let shuttingDown = false;

function log(message) {
  process.stdout.write(`[deploy] ${message}\n`);
}

function fail(message, code = 1) {
  process.stderr.write(`[deploy] ERROR: ${message}\n`);
  process.exit(code);
}

function ensureProjectRoot() {
  if (!fs.existsSync(path.join(projectRoot, "package.json"))) {
    fail(`package.json not found at ${projectRoot}`);
  }
  if (!fs.existsSync(path.join(projectRoot, "server", "src", "main.py"))) {
    fail("server/src/main.py not found");
  }
  if (!fs.existsSync(path.join(projectRoot, "client", "package.json"))) {
    fail("client/package.json not found");
  }
}

function ensurePython() {
  if (!fs.existsSync(pythonBin)) {
    fail(`Python venv not found: ${pythonBin}. Create server/.venv first.`);
  }
}

function ensureRuntimeDir() {
  fs.mkdirSync(runtimeDir, { recursive: true });
}

function run(command, args, options = {}) {
  log(`$ ${command} ${args.join(" ")}`);
  const result = spawnSync(command, args, {
    cwd: options.cwd || projectRoot,
    env: options.env || process.env,
    stdio: "inherit",
    shell: false,
  });

  if (result.error) {
    fail(result.error.message);
  }
  if (result.status !== 0) {
    fail(`Command failed with code ${result.status ?? 1}: ${command} ${args.join(" ")}`);
  }
}

function spawnLogged(name, command, args, options = {}) {
  log(`$ ${command} ${args.join(" ")}`);
  const child = spawn(command, args, {
    cwd: options.cwd || projectRoot,
    env: options.env || process.env,
    stdio: options.stdio || "inherit",
    shell: false,
  });

  child.on("error", (error) => {
    if (!shuttingDown) {
      process.stderr.write(`[deploy] ${name} failed: ${error.message}\n`);
      void stopLocalProd(1);
    }
  });

  child.on("exit", (code, signal) => {
    if (!shuttingDown && signal === "SIGTERM") {
      void stopLocalProd(0);
      return;
    }
    if (!shuttingDown && code !== 0 && options.failOnExit !== false) {
      const reason = signal ? `signal ${signal}` : `code ${code ?? 0}`;
      process.stderr.write(`[deploy] ${name} exited unexpectedly (${reason})\n`);
      void stopLocalProd(code || 1);
    }
  });

  return child;
}

function readPid(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8").trim();
  } catch {
    return "";
  }
}

function isPidRunning(pid) {
  if (!pid) return false;
  try {
    process.kill(Number(pid), 0);
    return true;
  } catch {
    return false;
  }
}

function killPid(pid) {
  if (!pid || !isPidRunning(pid)) return;
  if (isWindows) {
    spawnSync("taskkill", ["/pid", String(pid), "/T", "/F"], { stdio: "ignore" });
    return;
  }
  try {
    process.kill(Number(pid), "SIGTERM");
  } catch {}
}

async function fetchOk(url) {
  try {
    const response = await fetch(url);
    return response.ok;
  } catch {
    return false;
  }
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.text();
}

async function waitForBackend() {
  log("Waiting for backend health");
  for (let i = 0; i < 60; i += 1) {
    if (await fetchOk(`${backendUrl}/api/health`)) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  fail(`Backend did not become healthy at ${backendUrl}/api/health`);
}

function generateNginxConfig() {
  const templatePath = path.join(deployDir, "nginx", "local-test.conf.template");
  const outputPath = path.join(runtimeDir, "nginx.local-test.conf");
  const normalize = (value) => value.replaceAll("\\", "/");
  const content = fs
    .readFileSync(templatePath, "utf8")
    .replaceAll("__PROJECT_ROOT__", normalize(projectRoot))
    .replaceAll("__RUNTIME_DIR__", normalize(runtimeDir))
    .replaceAll("__BACKEND_PORT__", backendPort)
    .replaceAll("__FRONTEND_PORT__", frontendPort);

  fs.writeFileSync(outputPath, content);
  return outputPath;
}

async function initDb(mode = "migrate") {
  ensureProjectRoot();
  ensurePython();

  if (mode === "migrate") {
    run(npmBin, ["run", "db:migrate"]);
    return;
  }
  if (mode === "bootstrap") {
    run(npmBin, ["run", "db:bootstrap"]);
    return;
  }
  fail(`Unknown database mode: ${mode}. Use migrate or bootstrap.`);
}

function runDev() {
  ensureProjectRoot();
  ensurePython();
  run(process.execPath, [path.join(deployDir, "scripts", "run-local.js")]);
}

async function runLocalProd() {
  ensureProjectRoot();
  ensurePython();
  ensureRuntimeDir();

  if (process.env.RUN_MIGRATIONS !== "0") {
    await initDb("migrate");
  }

  if (process.env.SKIP_BUILD !== "1") {
    run(npmBin, ["--prefix", "client", "run", "build"]);
  }

  const indexPath = path.join(projectRoot, "client", "dist", "index.html");
  if (!fs.existsSync(indexPath)) {
    fail("client/dist/index.html not found. Run npm --prefix client run build first.");
  }

  const nginxConfig = generateNginxConfig();
  log(`Generated nginx config: ${nginxConfig}`);
  run("nginx", ["-t", "-c", nginxConfig]);

  const backendPidPath = path.join(runtimeDir, "backend.pid");
  const existingPid = readPid(backendPidPath);
  if (isPidRunning(existingPid)) {
    fail(`Backend already running with pid ${existingPid}. Stop it with npm run deploy:prod:stop.`);
  }

  const backendLog = fs.openSync(path.join(runtimeDir, "backend.log"), "a");
  log(`Starting backend at ${backendUrl}`);
  backendProcess = spawnLogged(
    "backend",
    pythonBin,
    ["src/main.py"],
    {
      cwd: path.join(projectRoot, "server"),
      env: {
        ...process.env,
        DEBUG: "false",
        SERVER_HOST: backendHost,
        SERVER_PORT: backendPort,
        SERVER_RELOAD: "false",
      },
      stdio: ["ignore", backendLog, backendLog],
    }
  );
  fs.writeFileSync(backendPidPath, String(backendProcess.pid));

  await waitForBackend();

  log("Production-like local stack is ready");
  log(`Frontend: ${frontendUrl}`);
  log(`Backend:  ${backendUrl}`);
  log(`Logs:     ${runtimeDir}`);
  log("Press Ctrl+C to stop backend and nginx");

  nginxProcess = spawnLogged("nginx", "nginx", ["-c", nginxConfig, "-g", "daemon off;"]);

  process.on("SIGINT", () => void stopLocalProd(0));
  process.on("SIGTERM", () => void stopLocalProd(0));
}

async function checkLocalProd() {
  ensureRuntimeDir();

  log("Checking frontend root");
  await fetchText(`${frontendUrl}/`);

  log("Checking SPA fallback");
  await fetchText(`${frontendUrl}/groups`);

  log("Checking API proxy");
  const health = await fetchText(`${frontendUrl}/api/health`);
  if (!health.includes('"status":"OK"')) {
    fail(`Unexpected health response: ${health}`);
  }

  log("Local production checks passed");
}

async function stopLocalProd(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;

  const nginxConfig = path.join(runtimeDir, "nginx.local-test.conf");
  if (fs.existsSync(nginxConfig)) {
    spawnSync("nginx", ["-s", "stop", "-c", nginxConfig], { stdio: "ignore" });
  }

  const backendPidPath = path.join(runtimeDir, "backend.pid");
  const pid = readPid(backendPidPath);
  if (backendProcess && backendProcess.pid) {
    killPid(backendProcess.pid);
  } else {
    killPid(pid);
  }
  try {
    fs.rmSync(backendPidPath);
  } catch {}

  log("Local production stack stopped");
  process.exit(code);
}

async function main() {
  const command = process.argv[2] || "help";
  const arg = process.argv[3];

  if (command === "init-db") {
    await initDb(arg || "migrate");
  } else if (command === "dev") {
    runDev();
  } else if (command === "prod-local") {
    await runLocalProd();
  } else if (command === "prod-check") {
    await checkLocalProd();
  } else if (command === "prod-stop") {
    await stopLocalProd(0);
  } else {
    process.stdout.write(`Usage:
  node deploy/scripts/deploy.mjs init-db [migrate|bootstrap]
  node deploy/scripts/deploy.mjs dev
  node deploy/scripts/deploy.mjs prod-local
  node deploy/scripts/deploy.mjs prod-check
  node deploy/scripts/deploy.mjs prod-stop
`);
    process.exit(command === "help" ? 0 : 1);
  }
}

main().catch((error) => {
  fail(error instanceof Error ? error.message : String(error));
});
