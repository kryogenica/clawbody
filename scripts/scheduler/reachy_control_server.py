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
import urllib.error
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
            self._send(200, PAGE_HTML.encode(), "text/html; charset=utf-8")
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


# --------------------------------------------------------------------------- #
# Web page
# --------------------------------------------------------------------------- #
PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Reachy Mini · Control Center</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2330; --border:#30363d;
    --text:#e6edf3; --muted:#8b949e; --accent:#2f81f7;
    --on:#3fb950; --off:#f85149; --warn:#d29922;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg);color:var(--text);line-height:1.5}
  .wrap{max-width:880px;margin:0 auto;padding:24px 16px 64px}
  h1{font-size:22px;margin:0 0 4px;display:flex;align-items:center;gap:10px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:20px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;
    padding:18px;margin-bottom:18px}
  .row{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
  .dot{width:11px;height:11px;border-radius:50%;display:inline-block}
  .dot.on{background:var(--on);box-shadow:0 0 8px var(--on)}
  .dot.off{background:var(--off)}
  .dot.warn{background:var(--warn)}
  .stat{flex:1;min-width:150px;background:var(--panel2);border:1px solid var(--border);
    border-radius:10px;padding:12px 14px}
  .stat .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
  .stat .v{font-size:16px;font-weight:600;margin-top:3px}
  button{font:inherit;border:0;border-radius:9px;padding:11px 18px;cursor:pointer;
    font-weight:600;color:#fff;transition:.12s}
  button:disabled{opacity:.5;cursor:not-allowed}
  .btn-on{background:var(--on)} .btn-on:hover{filter:brightness(1.08)}
  .btn-off{background:var(--off)} .btn-off:hover{filter:brightness(1.08)}
  .btn-ghost{background:var(--panel2);border:1px solid var(--border)}
  .btn-sm{padding:6px 12px;font-size:13px;font-weight:500}
  label{font-size:13px;color:var(--muted);display:block;margin-bottom:4px}
  input,select{font:inherit;background:var(--panel2);border:1px solid var(--border);
    color:var(--text);border-radius:8px;padding:9px 10px;width:100%}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  h2{font-size:15px;margin:0 0 14px;color:var(--text)}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--border)}
  th{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
  .tag{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
  .tag.on{background:rgba(63,185,80,.15);color:var(--on)}
  .tag.off{background:rgba(248,81,73,.15);color:var(--off)}
  .tag.muted{background:var(--panel2);color:var(--muted)}
  .log{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
    color:var(--muted);max-height:170px;overflow:auto;white-space:pre-wrap}
  .toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);
    background:var(--panel2);border:1px solid var(--border);padding:10px 16px;
    border-radius:9px;opacity:0;transition:.2s;pointer-events:none}
  .toast.show{opacity:1}
  .muted{color:var(--muted)}
</style>
</head>
<body>
<div class="wrap">
  <h1><span id="power-dot" class="dot off"></span> Reachy Mini · Control Center</h1>
  <div class="sub" id="clock">—</div>

  <div class="card">
    <div class="grid" style="margin-bottom:16px">
      <div class="stat"><div class="k">Backend</div><div class="v" id="s-daemon">—</div></div>
      <div class="stat"><div class="k">Running app</div><div class="v" id="s-app">—</div></div>
      <div class="stat"><div class="k">Control bus</div><div class="v" id="s-bus">—</div></div>
      <div class="stat"><div class="k">App UI :7860</div><div class="v" id="s-ui">—</div></div>
    </div>
    <div class="row">
      <div style="flex:1;min-width:180px">
        <label>App to launch on power-on</label>
        <input id="app-name" value="reachy_mini_conversation_app"/>
      </div>
      <div style="display:flex;gap:10px;align-self:flex-end">
        <button class="btn-on" id="btn-on">Power ON</button>
        <button class="btn-off" id="btn-off">Power OFF</button>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Add timer</h2>
    <div class="grid">
      <div>
        <label>Action</label>
        <select id="t-action"><option value="on">Turn ON</option><option value="off">Turn OFF</option></select>
      </div>
      <div>
        <label>Repeat</label>
        <select id="t-kind"><option value="daily">Daily</option><option value="once">Once</option></select>
      </div>
      <div>
        <label id="t-time-label">Time (HH:MM)</label>
        <input id="t-time" placeholder="14:30"/>
      </div>
      <div>
        <label>Label (optional)</label>
        <input id="t-label" placeholder="e.g. Morning greet"/>
      </div>
    </div>
    <div class="row" style="margin-top:12px">
      <div id="t-app-wrap" style="flex:1;min-width:200px">
        <label>App (for ON timers)</label>
        <input id="t-app" value="reachy_mini_conversation_app"/>
      </div>
      <button class="btn-ghost" id="btn-add" style="align-self:flex-end">Add timer</button>
    </div>
  </div>

  <div class="card">
    <h2>Scheduled timers</h2>
    <table>
      <thead><tr><th>Action</th><th>When</th><th>App</th><th>Status</th><th></th></tr></thead>
      <tbody id="timer-rows"><tr><td colspan="5" class="muted">Loading…</td></tr></tbody>
    </table>
  </div>

  <div class="card">
    <h2>Activity log</h2>
    <div class="log" id="log">—</div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const $ = id => document.getElementById(id);
function toast(msg){const t=$("toast");t.textContent=msg;t.classList.add("show");
  setTimeout(()=>t.classList.remove("show"),2200);}
async function api(path, method="GET", body){
  const opt={method,headers:{"Content-Type":"application/json"}};
  if(body) opt.body=JSON.stringify(body);
  const r=await fetch(path,opt); return r.json().catch(()=>({}));
}

function setDot(el, up){el.className="dot "+(up?"on":"off");}

async function refreshStatus(){
  let s; try{ s=await api("/api/status"); }catch(e){ return; }
  $("clock").textContent = s.now + (s.busy ? "  ·  working…" : "");
  $("s-daemon").textContent = s.daemon_state || "unreachable";
  const app = s.app && s.app.info ? (s.app.info.name + " ("+(s.app.state)+")") : "none";
  $("s-app").textContent = app;
  $("s-bus").textContent = s.control_bus_up ? "up" : "down";
  $("s-ui").textContent = s.app_ui_up ? "up" : "down";
  const live = s.daemon_state==="running" && s.control_bus_up;
  $("power-dot").className = "dot "+(live?"on":(s.busy?"warn":"off"));
  $("btn-on").disabled = s.busy; $("btn-off").disabled = s.busy;
  $("log").textContent = (s.log||[]).map(l=>l.ts+"  "+l.msg).join("\n") || "—";
}

async function refreshTimers(){
  const d=await api("/api/timers"); const rows=$("timer-rows");
  if(!d.timers || !d.timers.length){rows.innerHTML='<tr><td colspan="5" class="muted">No timers yet.</td></tr>';return;}
  rows.innerHTML="";
  for(const t of d.timers){
    const tr=document.createElement("tr");
    const when = t.kind==="daily" ? ("daily "+t.time) : ("once "+t.time.replace("T"," "));
    let status = t.enabled ? '<span class="tag '+(t.action)+'">enabled</span>'
                           : '<span class="tag muted">'+(t.fired?"fired":"off")+'</span>';
    tr.innerHTML =
      '<td><span class="tag '+t.action+'">'+(t.action==="on"?"ON":"OFF")+'</span>'+
        (t.label?' '+t.label:'')+'</td>'+
      '<td>'+when+'</td>'+
      '<td class="muted">'+(t.action==="on"?t.app_name:"—")+'</td>'+
      '<td>'+status+'</td>'+
      '<td style="text-align:right;white-space:nowrap">'+
        '<button class="btn-ghost btn-sm" data-tg="'+t.id+'" data-en="'+(t.enabled?0:1)+'">'+
          (t.enabled?"Disable":"Enable")+'</button> '+
        '<button class="btn-off btn-sm" data-del="'+t.id+'">Delete</button>'+
      '</td>';
    rows.appendChild(tr);
  }
  rows.querySelectorAll("[data-del]").forEach(b=>b.onclick=async()=>{
    await api("/api/timers/delete","POST",{id:b.dataset.del}); refreshTimers();});
  rows.querySelectorAll("[data-tg]").forEach(b=>b.onclick=async()=>{
    await api("/api/timers/toggle","POST",{id:b.dataset.tg,enabled:b.dataset.en==="1"});refreshTimers();});
}

$("btn-on").onclick=async()=>{const r=await api("/api/power/on","POST",{app_name:$("app-name").value});
  toast(r.ok?"Powering on…":("Error: "+(r.error||"failed"))); setTimeout(refreshStatus,800);};
$("btn-off").onclick=async()=>{const r=await api("/api/power/off","POST",{});
  toast(r.ok?"Powering off…":("Error: "+(r.error||"failed"))); setTimeout(refreshStatus,800);};

$("t-kind").onchange=()=>{const daily=$("t-kind").value==="daily";
  $("t-time-label").textContent = daily?"Time (HH:MM)":"Date & time";
  $("t-time").placeholder = daily?"14:30":"2026-06-27T14:30";};
$("t-action").onchange=()=>{$("t-app-wrap").style.display = $("t-action").value==="on"?"block":"none";};

$("btn-add").onclick=async()=>{
  const body={action:$("t-action").value,kind:$("t-kind").value,time:$("t-time").value,
    label:$("t-label").value,app_name:$("t-app").value};
  const r=await api("/api/timers/add","POST",body);
  if(r.ok){toast("Timer added"); $("t-time").value=""; $("t-label").value=""; refreshTimers();}
  else toast("Error: "+(r.error||"failed"));
};

refreshStatus(); refreshTimers();
setInterval(refreshStatus,4000);
setInterval(refreshTimers,8000);
</script>
</body>
</html>
"""


def main() -> None:
    threading.Thread(target=scheduler_loop, daemon=True).start()
    httpd = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    log(f"control center on http://{LISTEN_HOST}:{LISTEN_PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
