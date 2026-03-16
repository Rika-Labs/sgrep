import { spawn, execSync, type ChildProcess } from "node:child_process";

let watchProcess: ChildProcess | null = null;

function isSgrepInstalled(): boolean {
  try {
    execSync("command -v sgrep", { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function runIndex(): void {
  try {
    execSync("sgrep index", { stdio: "ignore" });
  } catch {
    // Index failure is non-fatal
  }
}

function startWatch(): void {
  watchProcess = spawn("sgrep", ["watch"], {
    detached: true,
    stdio: "ignore",
  });
  watchProcess.unref();
}

function stopWatch(): void {
  if (watchProcess?.pid) {
    try {
      process.kill(watchProcess.pid);
    } catch {
      // Already exited
    }
    watchProcess = null;
  }
}

export function session_start(): void {
  if (!isSgrepInstalled()) {
    return;
  }
  runIndex();
  startWatch();
}

export function session_shutdown(): void {
  stopWatch();
}
