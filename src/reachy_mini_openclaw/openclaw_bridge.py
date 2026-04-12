"""ClawBody - Bridge to OpenClaw Gateway for AI responses.

This module provides ClawBody's integration with the OpenClaw gateway
using the WebSocket protocol (the gateway's native transport).

ClawBody uses OpenAI Realtime API for voice I/O (speech recognition + TTS)
but routes all responses through OpenClaw (Clawson) for intelligence.
"""

import json
import time
import asyncio
import logging
import uuid
from typing import Optional, Any, AsyncIterator
from dataclasses import dataclass

import websockets
from websockets.exceptions import ConnectionClosed

from reachy_mini_openclaw.config import config

logger = logging.getLogger(__name__)

# Protocol version supported by this client
PROTOCOL_VERSION = 3


@dataclass
class OpenClawResponse:
    """Response from OpenClaw gateway."""
    content: str
    error: Optional[str] = None


class OpenClawBridge:
    """Bridge to OpenClaw Gateway using WebSocket protocol.

    The OpenClaw gateway speaks WebSocket with a JSON frame protocol.
    This class handles the connect handshake, authentication, and
    chat operations.

    Example:
        bridge = OpenClawBridge()
        await bridge.connect()

        # Simple query
        response = await bridge.chat("Hello!")
        print(response.content)
    """

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        gateway_token: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """Initialize the OpenClaw bridge.

        Args:
            gateway_url: URL of the OpenClaw gateway (default: from env/config).
                         Accepts http:// or ws:// schemes; http is converted to ws.
            gateway_token: Authentication token (default: from env/config)
            agent_id: OpenClaw agent ID to use (default: from env/config)
            timeout: Request timeout in seconds
        """
        import os

        raw_url = (
            gateway_url
            or os.getenv("OPENCLAW_GATEWAY_URL")
            or config.OPENCLAW_GATEWAY_URL
        )
        # Normalise to ws:// (the gateway listens on the same port for both)
        self.gateway_url = self._normalise_ws_url(raw_url)

        self.gateway_token = (
            gateway_token
            or os.getenv("OPENCLAW_TOKEN")
            or config.OPENCLAW_TOKEN
        )
        self.agent_id = (
            agent_id
            or os.getenv("OPENCLAW_AGENT_ID")
            or config.OPENCLAW_AGENT_ID
        )
        self.timeout = timeout

        # Session key – "main" shares context with WhatsApp and other channels.
        # Full key format: agent:<agent_id>:<session_key>
        self.session_key = (
            os.getenv("OPENCLAW_SESSION_KEY")
            or config.OPENCLAW_SESSION_KEY
            or "main"
        )

        # Persistent WebSocket state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._conn_id: Optional[str] = None

        # Background listener task & pending request futures
        self._listener_task: Optional[asyncio.Task] = None
        self._pending: dict[str, asyncio.Future] = {}
        # Events keyed by runId -> list of event payloads
        self._run_events: dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_ws_url(url: str) -> str:
        """Convert http(s) URL to ws(s)."""
        if url.startswith("http://"):
            return "ws://" + url[7:]
        if url.startswith("https://"):
            return "wss://" + url[8:]
        if not url.startswith("ws://") and not url.startswith("wss://"):
            return "ws://" + url
        return url

    @staticmethod
    def _connect_error_is_pairing_pending(error: dict[str, Any]) -> bool:
        """True when the gateway rejected connect until an operator approves device pairing."""
        code = str(error.get("code") or "").upper()
        msg = str(error.get("message") or "").lower()
        details = error.get("details") or {}
        if not isinstance(details, dict):
            details = {}
        dcode = str(details.get("code") or "").upper()
        reason = str(details.get("reason") or "").lower()
        if code == "NOT_PAIRED" or dcode == "PAIRING_REQUIRED":
            return True
        if "pairing" in msg or "not paired" in msg or "pairing" in reason:
            return True
        return False

    @staticmethod
    def _connection_closed_is_pairing(exc: BaseException) -> bool:
        if not isinstance(exc, ConnectionClosed):
            return False
        reason = (exc.reason or "").lower()
        return "pairing" in reason

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the OpenClaw gateway and authenticate.

        When device pairing is required (typical for LAN), the gateway closes the
        socket with ``NOT_PAIRED`` / "pairing required". We poll and retry until
        an operator approves in the gateway UI (or timeout).

        Returns:
            True if connection successful, False otherwise
        """
        import os

        wait_sec = float(os.environ.get("OPENCLAW_PAIRING_WAIT_SEC", "300"))
        poll_sec = float(os.environ.get("OPENCLAW_PAIRING_POLL_SEC", "2"))
        deadline = time.monotonic() + max(5.0, wait_sec)
        attempt = 0

        logger.info(
            "Connecting to OpenClaw at %s (token: %s)",
            self.gateway_url,
            "set" if self.gateway_token else "not set",
        )

        while time.monotonic() < deadline:
            if attempt > 0:
                delay = min(poll_sec, max(0.25, deadline - time.monotonic()))
                await asyncio.sleep(delay)
            attempt += 1
            status = await self._connect_handshake_attempt()
            if status == "connected":
                return True
            if status == "pairing_required":
                left = max(0.0, deadline - time.monotonic())
                logger.info(
                    "OpenClaw device pairing still pending — approve ClawBody (Reachy Mini) in the "
                    "gateway (same requestId in logs means the gateway has not cleared that request "
                    "yet). Retrying every %.0fs for up to %.0fs more (this is a client retry budget, "
                    "not a countdown to approve).",
                    poll_sec,
                    left,
                )
                continue
            return False

        logger.error(
            "OpenClaw connect timed out after %.0fs waiting for device pairing approval",
            wait_sec,
        )
        await self._close_ws()
        return False

    async def _recv_until_connect_challenge(self, deadline: float) -> dict[str, Any]:
        """Read frames until ``connect.challenge`` (gateways may emit other events first)."""
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(
                self._ws.recv(),
                timeout=max(0.25, deadline - time.monotonic()),
            )
            msg = json.loads(raw)
            if msg.get("type") == "event" and msg.get("event") == "connect.challenge":
                return msg
            logger.debug(
                "OpenClaw ignoring pre-challenge frame type=%s event=%s",
                msg.get("type"),
                msg.get("event"),
            )
        raise TimeoutError("connect.challenge not received")

    async def _recv_until_connect_result(self, req_id: str, deadline: float) -> dict[str, Any]:
        """Read frames until the ``connect`` response for ``req_id``."""
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=max(0.25, deadline - time.monotonic()),
                )
            except ConnectionClosed:
                raise
            msg = json.loads(raw)
            if isinstance(msg, dict) and msg.get("type") == "res" and msg.get("id") == req_id:
                return msg
            if msg.get("type") == "event" and msg.get("event") == "device.pair.resolved":
                pl = msg.get("payload") or {}
                logger.info(
                    "OpenClaw device.pair.resolved decision=%s deviceId=%s requestId=%s",
                    pl.get("decision"),
                    pl.get("deviceId"),
                    pl.get("requestId"),
                )
                continue
            logger.debug(
                "OpenClaw while awaiting connect res: type=%s id=%s",
                msg.get("type"),
                msg.get("id"),
            )
        raise TimeoutError("connect response not received")

    async def _connect_handshake_attempt(self) -> str:
        """One WebSocket open + connect handshake. Returns ``connected``, ``pairing_required``, or ``failed``."""
        from reachy_mini_openclaw import openclaw_device_identity as odi

        await self._close_ws()
        try:
            origin = self.gateway_url.replace("ws://", "http://").replace("wss://", "https://")
            self._ws = await websockets.connect(
                self.gateway_url,
                origin=origin,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=5,
            )

            chal_deadline = time.monotonic() + 15.0
            try:
                challenge = await self._recv_until_connect_challenge(chal_deadline)
            except TimeoutError:
                logger.error("Timed out waiting for OpenClaw connect.challenge")
                await self._close_ws()
                return "failed"

            challenge_payload = challenge.get("payload") or {}
            nonce = challenge_payload.get("nonce")
            if not nonce:
                logger.error("OpenClaw connect.challenge missing nonce; cannot sign device identity")
                await self._close_ws()
                return "failed"

            req_id = str(uuid.uuid4())
            scopes = [
                "operator.read",
                "operator.write",
                "operator.admin",
            ]
            client_block = {
                "id": "gateway-client",
                "version": "1.0.0",
                "platform": "linux",
                "mode": "backend",
                "displayName": "ClawBody (Reachy Mini)",
                "deviceFamily": "reachy_mini",
            }
            auth: dict[str, str] = {}
            if self.gateway_token:
                auth["token"] = self.gateway_token
            id_path = odi.default_identity_path()
            extra_dt = (config.OPENCLAW_DEVICE_TOKEN or "").strip()
            if not extra_dt:
                extra_dt = (odi.load_stored_device_token(id_path) or "").strip()
            if extra_dt:
                auth["deviceToken"] = extra_dt

            params: dict[str, Any] = {
                "minProtocol": PROTOCOL_VERSION,
                "maxProtocol": PROTOCOL_VERSION,
                "client": client_block,
                "role": "operator",
                "scopes": scopes,
                "caps": [],
                "auth": auth,
                "locale": "en-US",
                "userAgent": "clawbody/reachy-mini",
            }

            if not odi.device_disabled():
                try:
                    priv, device_id = odi.load_or_create_identity(id_path)
                    logger.info(
                        "OpenClaw device identity deviceId=%s… (identity file %s)",
                        device_id[:12],
                        id_path,
                    )
                    signed_at_ms = int(time.time() * 1000)
                    # v3 payload must use the same credential string as the gateway's
                    # resolveSignatureToken: shared token first, else device token.
                    sig_token = auth.get("token") or auth.get("deviceToken") or ""
                    payload_v3 = odi.build_device_auth_payload_v3(
                        device_id=device_id,
                        client_id=client_block["id"],
                        client_mode=client_block["mode"],
                        role="operator",
                        scopes=scopes,
                        signed_at_ms=signed_at_ms,
                        token=sig_token,
                        nonce=str(nonce),
                        platform=client_block.get("platform"),
                        device_family=client_block.get("deviceFamily"),
                    )
                    signature = odi.sign_device_payload(priv, payload_v3)
                    public_key_b64url = odi.public_key_raw_b64url(priv)
                    params["device"] = {
                        "id": device_id,
                        "publicKey": public_key_b64url,
                        "signature": signature,
                        "signedAt": signed_at_ms,
                        "nonce": str(nonce),
                    }
                except Exception as e:
                    logger.error(
                        "OpenClaw device identity failed (%s). Install cryptography>=42 "
                        "or set OPENCLAW_DISABLE_DEVICE_IDENTITY=1 for legacy gateways.",
                        e,
                    )
                    await self._close_ws()
                    return "failed"
            else:
                logger.warning(
                    "OPENCLAW_DISABLE_DEVICE_IDENTITY is set; connect may lack operator scopes on modern gateways"
                )

            connect_req = {"type": "req", "id": req_id, "method": "connect", "params": params}
            await self._ws.send(json.dumps(connect_req))

            res_deadline = time.monotonic() + 20.0
            try:
                hello = await self._recv_until_connect_result(req_id, res_deadline)
            except ConnectionClosed as e:
                if self._connection_closed_is_pairing(e):
                    logger.warning("OpenClaw closed socket before connect res: %s", e.reason)
                    await self._close_ws()
                    return "pairing_required"
                raise
            except TimeoutError:
                logger.error("Timed out waiting for OpenClaw connect response")
                await self._close_ws()
                return "failed"

            if hello.get("ok"):
                self._connected = True
                payload = hello.get("payload", {})
                server = payload.get("server", {})
                self._conn_id = server.get("connId")
                hello_auth = payload.get("auth") or {}
                granted = (
                    payload.get("scopes")
                    or payload.get("grantedScopes")
                    or hello_auth.get("scopes")
                )
                if granted is not None:
                    logger.info("OpenClaw granted scopes: %s", granted)
                dtok = hello_auth.get("deviceToken")
                if dtok and not odi.device_disabled():
                    odi.maybe_store_device_token(odi.default_identity_path(), dtok)
                logger.info(
                    "Connected to OpenClaw gateway (server=%s, connId=%s)",
                    server.get("host", "?"),
                    self._conn_id,
                )
                self._listener_task = asyncio.create_task(
                    self._listen_loop(), name="openclaw-ws-listener"
                )
                return "connected"

            err = hello.get("error", {}) or {}
            if self._connect_error_is_pairing_pending(err):
                details = err.get("details") if isinstance(err.get("details"), dict) else {}
                rid = details.get("requestId") if isinstance(details, dict) else None
                logger.warning(
                    "OpenClaw connect: pairing required (%s) — approve this device in the gateway "
                    "(requestId=%s). If you already approved: (1) OPENCLAW_TOKEN on the robot must "
                    "match gateway.auth.token in openclaw.json, not gateway.remote.token; "
                    "(2) approve the pending row whose deviceId matches the identity log line above; "
                    "(3) remove stale lastDeviceToken from the identity JSON or clear "
                    "OPENCLAW_DEVICE_TOKEN if reconnect still fails; (4) on the gateway host run "
                    "`openclaw devices list` / `openclaw devices approve` if the UI did not persist. "
                    "details=%s",
                    err.get("code"),
                    rid or "?",
                    err.get("details"),
                )
                await self._close_ws()
                return "pairing_required"

            logger.error(
                "OpenClaw connect failed: %s - %s details=%s",
                err.get("code"),
                err.get("message"),
                err.get("details"),
            )
            await self._close_ws()
            return "failed"

        except ConnectionClosed as e:
            if self._connection_closed_is_pairing(e):
                logger.warning("OpenClaw WebSocket closed during handshake: %s", e.reason)
                await self._close_ws()
                return "pairing_required"
            logger.error("OpenClaw WebSocket closed during handshake: %s", e)
            await self._close_ws()
            return "failed"
        except Exception as e:
            logger.error(
                "Failed to connect to OpenClaw gateway: %s (%s)",
                e,
                type(e).__name__,
            )
            await self._close_ws()
            return "failed"

    async def disconnect(self) -> None:
        """Disconnect from the gateway."""
        self._connected = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._close_ws()

    async def _close_ws(self) -> None:
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ------------------------------------------------------------------
    # Background listener
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Background task that reads all frames from the WebSocket."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._dispatch(msg)
        except websockets.ConnectionClosed as e:
            logger.warning("OpenClaw WebSocket closed: %s", e)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("OpenClaw listener error: %s", e)
        finally:
            self._connected = False

    async def _dispatch(self, msg: dict) -> None:
        """Route an incoming frame to the right handler."""
        msg_type = msg.get("type")

        if msg_type == "res":
            # Response to a request we sent
            req_id = msg.get("id")
            fut = self._pending.pop(req_id, None)
            if fut and not fut.done():
                fut.set_result(msg)

        elif msg_type == "event":
            event_name = msg.get("event", "")
            payload = msg.get("payload", {})

            # Route agent / chat events to the correct run queue
            run_id = payload.get("runId")
            if run_id and run_id in self._run_events:
                await self._run_events[run_id].put(msg)

            # Ignore noisy events silently
            if event_name in ("health", "tick"):
                return

            logger.debug("Event: %s (runId=%s)", event_name, run_id)

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    async def _send_request(
        self, method: str, params: dict, timeout: Optional[float] = None
    ) -> dict:
        """Send a request and wait for the response.

        Args:
            method: The RPC method name
            params: The params dict
            timeout: Override timeout (defaults to self.timeout)

        Returns:
            The full response message dict
        """
        if not self._ws or not self._connected:
            return {"ok": False, "error": {"code": "NOT_CONNECTED", "message": "Not connected"}}

        req_id = str(uuid.uuid4())
        req = {"type": "req", "id": req_id, "method": method, "params": params}

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        try:
            await self._ws.send(json.dumps(req))
            result = await asyncio.wait_for(fut, timeout=timeout or self.timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": {"code": "TIMEOUT", "message": "Request timed out"}}
        except Exception as e:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": {"code": "ERROR", "message": str(e)}}

    def _full_session_key(self) -> str:
        """Build the full session key: agent:<agentId>:<sessionKey>."""
        return f"agent:{self.agent_id}:{self.session_key}"

    # ------------------------------------------------------------------
    # Chat API
    # ------------------------------------------------------------------

    async def chat(
        self,
        message: str,
        image_b64: Optional[str] = None,
        system_context: Optional[str] = None,
    ) -> OpenClawResponse:
        """Send a message to OpenClaw and get a response.

        OpenClaw maintains conversation memory on its end, so it will be aware
        of conversations from other channels (WhatsApp, web, etc.). We only send
        the current message and let OpenClaw handle the context.

        Args:
            message: The user's message (transcribed speech)
            image_b64: Optional base64-encoded image from robot camera (not yet
                       supported over WebSocket chat.send – reserved for future)
            system_context: Optional additional system context (prepended to message)

        Returns:
            OpenClawResponse with the AI's response
        """
        if not self._connected:
            return OpenClawResponse(content="", error="Not connected to OpenClaw")

        # Prefix system context if provided
        final_message = message
        if system_context:
            final_message = f"[System: {system_context}]\n\n{message}"

        # If image provided, mention it (WebSocket protocol uses string messages;
        # image passing would require a separate mechanism)
        if image_b64:
            final_message = f"[Image attached]\n{final_message}"

        idempotency_key = str(uuid.uuid4())
        session_key = self._full_session_key()

        # Create a queue to collect events for this run
        # We'll get the runId from the response
        params = {
            "idempotencyKey": idempotency_key,
            "sessionKey": session_key,
            "message": final_message,
        }

        try:
            # Send the request
            resp = await self._send_request("chat.send", params, timeout=30)

            if not resp.get("ok"):
                err = resp.get("error", {})
                error_msg = f"{err.get('code', 'UNKNOWN')}: {err.get('message', 'Unknown error')}"
                logger.error("chat.send failed: %s", error_msg)
                return OpenClawResponse(content="", error=error_msg)

            run_id = resp.get("payload", {}).get("runId")
            if not run_id:
                return OpenClawResponse(content="", error="No runId in response")

            # Register a queue to receive events for this run
            event_queue: asyncio.Queue = asyncio.Queue()
            self._run_events[run_id] = event_queue

            try:
                # Collect the streamed response
                full_text = ""
                while True:
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(), timeout=self.timeout
                        )
                        payload = event.get("payload", {})
                        event_name = event.get("event", "")

                        if event_name == "agent":
                            stream = payload.get("stream")
                            data = payload.get("data", {})

                            if stream == "assistant":
                                # Accumulate the full text
                                full_text = data.get("text", full_text)

                            elif stream == "lifecycle" and data.get("phase") == "end":
                                # Run completed
                                break

                        elif event_name == "chat":
                            state = payload.get("state")
                            if state == "final":
                                # Extract final text
                                msg_payload = payload.get("message", {})
                                content_parts = msg_payload.get("content", [])
                                if isinstance(content_parts, list):
                                    for part in content_parts:
                                        if isinstance(part, dict) and part.get("type") == "text":
                                            full_text = part.get("text", full_text)
                                elif isinstance(content_parts, str):
                                    full_text = content_parts
                                break

                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for chat response (runId=%s)", run_id)
                        if full_text:
                            break
                        return OpenClawResponse(content="", error="Response timeout")

                return OpenClawResponse(content=full_text)

            finally:
                self._run_events.pop(run_id, None)

        except Exception as e:
            logger.error("OpenClaw chat error: %s", e)
            return OpenClawResponse(content="", error=str(e))

    async def stream_chat(
        self,
        message: str,
        image_b64: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a response from OpenClaw.

        Args:
            message: The user's message
            image_b64: Optional base64-encoded image

        Yields:
            String chunks of the response as they arrive
        """
        if not self._connected:
            yield "[Error: Not connected to OpenClaw]"
            return

        final_message = message
        if image_b64:
            final_message = f"[Image attached]\n{message}"

        params = {
            "idempotencyKey": str(uuid.uuid4()),
            "sessionKey": self._full_session_key(),
            "message": final_message,
        }

        try:
            resp = await self._send_request("chat.send", params, timeout=30)

            if not resp.get("ok"):
                err = resp.get("error", {})
                yield f"[Error: {err.get('message', 'Unknown error')}]"
                return

            run_id = resp.get("payload", {}).get("runId")
            if not run_id:
                yield "[Error: No runId]"
                return

            event_queue: asyncio.Queue = asyncio.Queue()
            self._run_events[run_id] = event_queue

            try:
                prev_text = ""
                while True:
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(), timeout=self.timeout
                        )
                        payload = event.get("payload", {})
                        event_name = event.get("event", "")

                        if event_name == "agent":
                            stream = payload.get("stream")
                            data = payload.get("data", {})

                            if stream == "assistant":
                                delta = data.get("delta", "")
                                if delta:
                                    yield delta

                            elif stream == "lifecycle" and data.get("phase") == "end":
                                break

                        elif event_name == "chat" and payload.get("state") == "final":
                            break

                    except asyncio.TimeoutError:
                        yield "[Error: timeout]"
                        break
            finally:
                self._run_events.pop(run_id, None)

        except Exception as e:
            logger.error("OpenClaw streaming error: %s", e)
            yield f"[Error: {e}]"

    @property
    def is_connected(self) -> bool:
        """Check if bridge is connected to gateway."""
        return self._connected

    async def get_agent_context(self) -> Optional[str]:
        """Fetch the agent's current context, personality, and memory summary.

        This asks OpenClaw to provide a summary of:
        - The agent's personality and identity
        - Recent conversation context
        - Important memories about the user
        - Current state

        Returns:
            A context string to use as system instructions, or None if failed
        """
        try:
            response = await self.chat(
                message="Provide your current context summary for the robot body.",
                system_context=(
                    "You are being asked to provide your current context for your robot body. "
                    "Output a comprehensive context summary that another AI can use to embody you. Include: "
                    "1. YOUR IDENTITY: Who you are, your name, your personality traits, how you speak. "
                    "2. USER CONTEXT: What you know about the user (name, preferences, relationship). "
                    "3. RECENT CONTEXT: Summary of recent conversations or important ongoing topics. "
                    "4. MEMORIES: Key things you remember that are relevant to interactions. "
                    "5. CURRENT STATE: Any relevant time/date awareness, ongoing tasks. "
                    "Be specific and personal. This context will be used by your robot body to speak and act AS YOU. "
                    "Output ONLY the context summary, no preamble."
                ),
            )

            if response.error:
                logger.warning("Failed to get agent context: %s", response.error)
                return None

            if response.content:
                logger.info(
                    "Retrieved agent context from OpenClaw (%d chars)",
                    len(response.content),
                )
                return response.content

            logger.warning("No context returned from OpenClaw")
            return None

        except Exception as e:
            logger.error("Failed to get agent context: %s", e)
            return None

    async def sync_conversation(
        self, user_message: str, assistant_response: str
    ) -> None:
        """Sync a conversation turn back to OpenClaw for memory continuity.

        Args:
            user_message: What the user said
            assistant_response: What the robot/AI responded
        """
        try:
            await self.chat(
                message=(
                    f"[ROBOT BODY SYNC] The following happened through the Reachy Mini robot:\n"
                    f"User said: {user_message}\n"
                    f"You responded: {assistant_response}\n"
                    f"Remember this as part of your ongoing conversation."
                ),
                system_context=(
                    "[ROBOT BODY SYNC] The following conversation happened through your "
                    "Reachy Mini robot body. Remember it as part of your ongoing conversation "
                    "with the user."
                ),
            )
            logger.debug("Synced conversation to OpenClaw")
        except Exception as e:
            logger.debug("Failed to sync conversation: %s", e)


# Global bridge instance (lazy initialization)
_bridge: Optional[OpenClawBridge] = None


def get_bridge() -> OpenClawBridge:
    """Get the global OpenClaw bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = OpenClawBridge()
    return _bridge
