#!/usr/bin/env python3
"""Reachy Mini control center.

A small, dependency-free web app (stdlib only) that lets you:
  * See live robot status (daemon backend + currently running app)
  * Power the robot ON (wake + start an app) or OFF (stop app + sleep)
  * Create/delete scheduled timers (one-shot or daily) for ON and OFF

It drives the Reachy Mini daemon REST API on localhost:8000 (the same backend
the :8000 dashboard uses). An internal scheduler thread fires timers, so they
work whether or not anyone has the page open. Run it as a systemd service so it
survives reboots and SSH disconnects.

Listens on 0.0.0.0:1991 -> reachable at http://127.0.0.1:1991 (on the robot or
via SSH tunnel) and http://<robot-ip>:1991 on the LAN.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

DAEMON = "http://127.0.0.1:8000"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 1991
DEFAULT_APP = "reachy_mini_conversation_app"
STATE_FILE = Path(__file__).resolve().parent / "control_state.json"
HTML_FILE = Path(__file__).resolve().parent / "control_center.html"

# Robot control bus / app dashboard ports, used for liveness checks.
PORT_CONTROL_BUS = 7447
PORT_APP_UI = 7860

_state_lock = threading.Lock()  # guards the timers state file
_log_lock = threading.Lock()  # guards the in-memory activity log
_action_lock = threading.Lock()  # serialize power actions
_log: list[dict] = []
_busy = False  # a power action is in progress


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    # Uses its own lock (never _state_lock) so it is always safe to call from
    # inside scheduler/state code without risk of self-deadlock.
    entry = {"ts": datetime.now().strftime("%H:%M:%S"), "msg": msg}
    print(f"[control] {entry['ts']} {msg}", flush=True)
    with _log_lock:
        _log.append(entry)
        del _log[:-60]  # keep last 60 lines


# --------------------------------------------------------------------------- #
# State persistence (timers)
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {"timers": []}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


# --------------------------------------------------------------------------- #
# Daemon API helpers
# --------------------------------------------------------------------------- #
def daemon_request(method: str, path: str, timeout: float = 30.0):
    url = f"{DAEMON}{path}"
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def daemon_state() -> str | None:
    try:
        return daemon_request("GET", "/api/daemon/status", timeout=5).get("state")
    except Exception:  # noqa: BLE001
        return None


def current_app() -> dict | None:
    try:
        return daemon_request("GET", "/api/apps/current-app-status", timeout=5)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Power actions
# --------------------------------------------------------------------------- #
def power_on(app_name: str = DEFAULT_APP) -> None:
    """Wake the robot backend, then start the requested app."""
    global _busy
    with _action_lock:
        _busy = True
        try:
            log(f"POWER ON requested (app={app_name})")

            # Skip if the robot is already on (an app is up).
            existing = current_app()
            if existing and existing.get("info") and existing.get("state") in ("running", "starting"):
                running_name = existing.get("info", {}).get("name")
                if running_name == app_name:
                    log(f"already on: '{app_name}' is {existing.get('state')}; nothing to do")
                else:
                    log(f"already on: app '{running_name}' is {existing.get('state')}; leaving it (skip)")
                return

            if daemon_state() != "running":
                log("starting backend (wake_up=true)...")
                daemon_request("POST", "/api/daemon/start?wake_up=true", timeout=60)
            else:
                log("backend already running")

            # Wait for backend + control bus.
            for _ in range(40):
                if daemon_state() == "running" and port_open(PORT_CONTROL_BUS):
                    break
                time.sleep(1.5)
            else:
                log("ERROR: backend/control bus did not come up; aborting")
                return
            log("backend ready, control bus up")

            log(f"starting app {app_name}...")
            daemon_request("POST", f"/api/apps/start-app/{app_name}", timeout=30)
            for _ in range(20):
                app = current_app()
                if app and app.get("state") == "running":
                    log(f"app {app_name} is running")
                    return
                if app and app.get("state") == "error":
                    log(f"ERROR: app reported error: {app.get('error')}")
                    return
                time.sleep(1.5)
            log("app start issued (still initializing)")
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR during power on: {exc}")
        finally:
            _busy = False


def power_off() -> None:
    """Stop the current app, then put the robot backend to sleep."""
    global _busy
    with _action_lock:
        _busy = True
        try:
            log("POWER OFF requested")
            app = current_app()
            if app and app.get("info"):
                try:
                    daemon_request("POST", "/api/apps/stop-current-app", timeout=20)
                    log("app stopped")
                except Exception as exc:  # noqa: BLE001
                    log(f"app stop note: {exc}")
                time.sleep(1)
            else:
                log("no app running")
            if daemon_state() in ("running", "starting"):
                log("putting backend to sleep (goto_sleep=true)...")
                daemon_request("POST", "/api/daemon/stop?goto_sleep=true", timeout=40)
            log("power off complete")
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR during power off: {exc}")
        finally:
            _busy = False


def run_action(action: str, app_name: str = DEFAULT_APP) -> None:
    target = power_on if action == "on" else power_off
    args = (app_name,) if action == "on" else ()
    threading.Thread(target=target, args=args, daemon=True).start()


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
def scheduler_loop() -> None:
    log("scheduler started")
    while True:
        try:
            check_schedules()
        except Exception as exc:  # noqa: BLE001
            log(f"scheduler error: {exc}")
        time.sleep(15)


def check_schedules() -> None:
    # Re-read the system timezone each tick. The robot has no RTC, so at boot the
    # timezone may still be wrong (geo-timezone.service sets it after the network is
    # up). tzset() lets this long-running process pick up that change without a
    # restart, so timers always evaluate against the correct local wall-clock time.
    time.tzset()
    now = datetime.now()
    today = date.today().isoformat()
    hhmm = now.strftime("%H:%M")

    to_fire: list[dict] = []
    with _state_lock:
        state = load_state()
        timers = state.get("timers", [])
        changed = False

        for t in timers:
            if not t.get("enabled", True):
                continue
            fire = False
            if t["kind"] == "daily":
                if t["time"] == hhmm and t.get("last_run_date") != today:
                    fire = True
                    t["last_run_date"] = today
                    changed = True
            elif t["kind"] == "once":
                try:
                    target = datetime.fromisoformat(t["time"])
                except ValueError:
                    continue
                if now >= target and not t.get("fired"):
                    fire = True
                    t["fired"] = True
                    t["fired_at"] = now.isoformat(timespec="seconds")
                    t["enabled"] = False
                    changed = True

            if fire:
                to_fire.append({
                    "label": t.get("label") or t["id"][:8],
                    "action": t["action"],
                    "app_name": t.get("app_name", DEFAULT_APP),
                })

        if changed:
            save_state(state)

    # Fire actions AFTER releasing _state_lock so logging/power actions never
    # contend with (or re-enter) the state lock.
    for f in to_fire:
        log(f"timer '{f['label']}' firing: {f['action']}")
        run_action(f["action"], f["app_name"])


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "ReachyControl/1.0"

    def log_message(self, *args):  # silence default noisy logging
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # -- GET ---------------------------------------------------------------- #
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            try:
                body = HTML_FILE.read_bytes()
            except OSError as exc:
                log(f"ERROR: could not read {HTML_FILE.name}: {exc}")
                self._send(500, b"control_center.html not found", "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._json(self._status())
        elif self.path == "/api/timers":
            with _state_lock:
                self._json({"timers": load_state().get("timers", [])})
        else:
            self._json({"error": "not found"}, 404)

    # -- POST --------------------------------------------------------------- #
    def do_POST(self):
        body = self._read_body()
        if self.path == "/api/power/on":
            if _busy:
                self._json({"ok": False, "error": "busy"}, 409)
                return
            run_action("on", body.get("app_name") or DEFAULT_APP)
            self._json({"ok": True})
        elif self.path == "/api/power/off":
            if _busy:
                self._json({"ok": False, "error": "busy"}, 409)
                return
            run_action("off")
            self._json({"ok": True})
        elif self.path == "/api/timers/add":
            self._json(self._add_timer(body))
        elif self.path == "/api/timers/delete":
            self._delete_timer(body.get("id"))
            self._json({"ok": True})
        elif self.path == "/api/timers/toggle":
            self._toggle_timer(body.get("id"), bool(body.get("enabled")))
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    # -- helpers ------------------------------------------------------------ #
    def _status(self) -> dict:
        app = current_app()
        return {
            "now": datetime.now().strftime("%a %Y-%m-%d %H:%M:%S %Z").strip(),
            "daemon_state": daemon_state(),
            "app": app,
            "control_bus_up": port_open(PORT_CONTROL_BUS),
            "app_ui_up": port_open(PORT_APP_UI),
            "busy": _busy,
            "default_app": DEFAULT_APP,
            "log": list(reversed(_log[-15:])),
        }

    def _add_timer(self, body: dict) -> dict:
        action = body.get("action")
        kind = body.get("kind")
        tval = (body.get("time") or "").strip()
        if action not in ("on", "off") or kind not in ("once", "daily") or not tval:
            return {"ok": False, "error": "invalid timer fields"}
        if kind == "daily":
            try:
                datetime.strptime(tval, "%H:%M")
            except ValueError:
                return {"ok": False, "error": "daily time must be HH:MM"}
        else:
            try:
                datetime.fromisoformat(tval)
            except ValueError:
                return {"ok": False, "error": "once time must be YYYY-MM-DDTHH:MM"}
        timer = {
            "id": uuid4().hex,
            "action": action,
            "kind": kind,
            "time": tval,
            "app_name": body.get("app_name") or DEFAULT_APP,
            "label": (body.get("label") or "").strip(),
            "enabled": True,
        }
        with _state_lock:
            state = load_state()
            state.setdefault("timers", []).append(timer)
            save_state(state)
        log(f"timer added: {action} ({kind} {tval})")
        return {"ok": True, "timer": timer}

    def _delete_timer(self, tid: str | None) -> None:
        with _state_lock:
            state = load_state()
            state["timers"] = [t for t in state.get("timers", []) if t["id"] != tid]
            save_state(state)

    def _toggle_timer(self, tid: str | None, enabled: bool) -> None:
        with _state_lock:
            state = load_state()
            for t in state.get("timers", []):
                if t["id"] == tid:
                    t["enabled"] = enabled
                    if enabled and t["kind"] == "once":
                        t["fired"] = False
            save_state(state)


def migrate_state() -> None:
    """One-time move of the timers file from the old scripts/ location."""
    old = Path(__file__).resolve().parent.parent / "control_state.json"
    if old.exists() and not STATE_FILE.exists():
        try:
            old.replace(STATE_FILE)
            log(f"migrated timers state from {old} to {STATE_FILE}")
        except OSError as exc:
            log(f"WARNING: could not migrate state from {old}: {exc}")


def main() -> None:
    migrate_state()
    threading.Thread(target=scheduler_loop, daemon=True).start()
    httpd = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    log(f"control center on http://{LISTEN_HOST}:{LISTEN_PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
