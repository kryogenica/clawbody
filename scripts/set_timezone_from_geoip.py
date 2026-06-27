#!/usr/bin/env python3
"""Detect the robot's location via IP geolocation and set the system timezone.

The Reachy Mini has no battery-backed RTC, so the clock is driven by NTP once
the network is up. NTP fixes the *instant* (UTC), but not the local timezone.
This script figures out where the robot is from its public IP and applies the
matching IANA timezone with `timedatectl`, so local time is correct wherever
the robot is plugged in.

Designed to run at boot (after the network is online) via a systemd service.
Uses only the Python standard library so it can run with the system python3.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Geolocation providers, tried in order. Each returns an IANA timezone string.
PROVIDERS = (
    ("http://ip-api.com/json/?fields=status,message,timezone,city,country", "timezone"),
    ("https://ipapi.co/json/", "timezone"),
    ("https://ipinfo.io/json", "timezone"),
)

ZONEINFO = Path("/usr/share/zoneinfo")
HTTP_TIMEOUT = 8  # seconds per provider
MAX_ATTEMPTS = 6  # retries while the network settles after boot
RETRY_DELAY = 10  # seconds between attempts


def log(msg: str) -> None:
    print(f"[geo-timezone] {msg}", flush=True)


def fetch_timezone() -> str | None:
    """Query providers until one returns a valid IANA timezone."""
    for url, key in PROVIDERS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "reachy-mini-geo-tz/1.0"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                data = json.load(resp)
        except Exception as exc:  # noqa: BLE001 - any failure -> try next provider
            log(f"provider failed: {url} ({exc})")
            continue

        if isinstance(data, dict) and data.get("status") == "fail":
            log(f"provider error: {url} ({data.get('message')})")
            continue

        tz = data.get(key) if isinstance(data, dict) else None
        if tz and is_valid_timezone(tz):
            city = data.get("city", "?")
            country = data.get("country") or data.get("country_name", "?")
            log(f"detected {tz} (location: {city}, {country}) via {url}")
            return tz
        log(f"no valid timezone from {url} (got {tz!r})")
    return None


def is_valid_timezone(tz: str) -> bool:
    """Guard against path traversal and confirm the zoneinfo file exists."""
    if not tz or tz.startswith("/") or ".." in tz:
        return False
    return (ZONEINFO / tz).is_file()


def current_timezone() -> str | None:
    link = Path("/etc/localtime")
    try:
        target = os.readlink(link)
    except OSError:
        return None
    marker = "zoneinfo/"
    idx = target.find(marker)
    return target[idx + len(marker):] if idx != -1 else None


def set_timezone(tz: str) -> bool:
    try:
        subprocess.run(["timedatectl", "set-timezone", tz], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log(f"failed to set timezone: {exc}")
        return False
    log(f"timezone set to {tz}")
    return True


def main() -> int:
    desired = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        desired = fetch_timezone()
        if desired:
            break
        if attempt < MAX_ATTEMPTS:
            log(f"attempt {attempt}/{MAX_ATTEMPTS} failed; retrying in {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)

    if not desired:
        log("could not determine timezone from any provider; leaving clock unchanged")
        return 1

    current = current_timezone()
    if current == desired:
        log(f"timezone already correct ({current}); nothing to do")
        return 0

    log(f"updating timezone: {current or 'unknown'} -> {desired}")
    return 0 if set_timezone(desired) else 1


if __name__ == "__main__":
    sys.exit(main())
