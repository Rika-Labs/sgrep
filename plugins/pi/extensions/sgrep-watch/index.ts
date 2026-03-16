import type { ExtensionContext } from "@mariozechner/pi-coding-agent";
import { execSync } from "node:child_process";

let watchPid: number | null = null;

function isSgrepInstalled(): boolean {
  try {
    execSync("command -v sgrep", { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function startWatch(): void {
  try {
    execSync("sgrep index", { stdio: "ignore" });
  } catch {
    // Index failure is non-fatal
  }
  try {
    const output = execSync("sgrep watch --detach", { encoding: "utf-8" });
    const match = output.match(/\d+/);
    if (match) {
      watchPid = parseInt(match[0], 10);
    }
  } catch {
    // Watch failure is non-fatal
  }
}

function stopWatch(): void {
  if (watchPid !== null) {
    try {
      process.kill(watchPid);
    } catch {
      // Already exited
    }
    watchPid = null;
  }
}

export default function sgrepWatch(context: ExtensionContext) {
  if (!isSgrepInstalled()) {
    return {};
  }

  startWatch();

  return {
    shutdown() {
      stopWatch();
    },
  };
}
