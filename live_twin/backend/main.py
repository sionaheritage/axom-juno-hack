"""
FastAPI app: streams live pose landmarks to the frontend over WebSocket,
and fires TENS pads (through ActuationController — never TensClient
directly) when the frontend reports a selected motion.

WebSocket contract (my proposed default — confirm with whoever's doing
frontend, then delete this comment):
    Server -> client:
        {"type": "ready", "armed": bool}
            sent once on connect. armed=false means every fire is mocked
            (logged, not sent to real hardware) regardless of what the
            client asks for — the UI should show this state unmistakably.
        {"type": "pose",
         "landmarks": {"shoulder": [x,y,z], "elbow": [x,y,z], "wrist": [x,y,z]} | null,
         "side": "left" | "right" | null,
         "status": "tracking" | "no_person" | "arm_not_visible"}
            sent every frame, including frames with no landmarks — a client
            that stops receiving pose messages should treat that as the
            stream being down, not as "arm out of frame". status says which
            of those two it is so the UI can prompt the user specifically.
        {"type": "status", "pad": "BICEP", "state": "firing", "duration_ms": 800}
        {"type": "status", "pad": "BICEP", "state": "done" | "error", "detail": "..."}
        {"type": "side", "side": "left" | "right" | null}
            ack of a set_side request; null means auto-pick.
    Client -> server:
        {"type": "select_motion", "motion": "grip" | "raise" | "lower" | "push_forward" | "pull_back"}
        {"type": "set_side", "side": "left" | "right" | null}
            forces which arm is tracked. A forced side never falls back to
            the other arm — see estimator.ArmPoseEstimator.

Note the side is broadcaster-wide, not per-connection: there is one camera
and one estimator, so the last client to set it wins for everyone.
"""
import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from live_twin.backend import config
from live_twin.backend.actuation.controller import ActuationController, ActuationError
from live_twin.backend.pose.broadcaster import PoseBroadcaster
from live_twin.backend.placement.pipeline import PlacementError, compute_placement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

broadcaster = PoseBroadcaster()
controller = ActuationController()

LIVE_TWIN_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = LIVE_TWIN_ROOT / "frontend"
SITE_TEMPLATE_DIR = LIVE_TWIN_ROOT.parent / "templates"
templates = Jinja2Templates(directory=[FRONTEND_DIR, SITE_TEMPLATE_DIR])


class LiveRuntime:
    """Own the shared camera/model while at least one live client needs it."""

    def __init__(
        self,
        pose_broadcaster: PoseBroadcaster,
        actuation_controller: ActuationController,
    ):
        self.broadcaster = pose_broadcaster
        self.controller = actuation_controller
        self._active_clients = 0
        self._lock = asyncio.Lock()

    @property
    def active_clients(self) -> int:
        return self._active_clients

    async def acquire(self) -> None:
        async with self._lock:
            if self._active_clients == 0:
                try:
                    await self.broadcaster.start()
                except Exception:
                    await self.broadcaster.stop()
                    raise
            self._active_clients += 1

    async def release(self) -> None:
        async with self._lock:
            if self._active_clients == 0:
                return
            self._active_clients -= 1
            if self._active_clients == 0:
                await self.controller.stop()
                await self.broadcaster.stop()

    async def shutdown(self) -> None:
        async with self._lock:
            self._active_clients = 0
            await self.controller.stop()
            await self.broadcaster.stop()


runtime = LiveRuntime(broadcaster, controller)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        yield
    finally:
        await runtime.shutdown()


app = FastAPI(title="Axon Live Twin", lifespan=_lifespan)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.get("/")
async def live_twin_home(request: Request):
    """Render Live Twin beneath the shared Axon site header."""
    return templates.TemplateResponse(request=request, name="twin.html")


async def _mjpeg_stream():
    try:
        async for jpeg in broadcaster.frames():
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                + jpeg
                + b"\r\n"
            )
    finally:
        await runtime.release()


@app.get("/camera.mjpeg")
async def camera_stream_endpoint():
    """Preview the broadcaster's camera without opening a second device handle."""
    try:
        await runtime.acquire()
    except Exception as exc:
        logger.exception("Live camera runtime could not start.")
        raise HTTPException(
            status_code=503,
            detail="The live camera or pose model could not be started.",
        ) from exc
    return StreamingResponse(
        _mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


async def _forward_pose(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        payload = await queue.get()
        await websocket.send_text(json.dumps(payload))


async def _handle_client_messages(websocket: WebSocket):
    while True:
        raw = await websocket.receive_text()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if msg.get("type") == "set_side":
            side = msg.get("side")
            try:
                broadcaster.set_side(side)
            except ValueError as exc:
                await websocket.send_text(json.dumps({
                    "type": "status", "pad": None, "state": "error", "detail": str(exc),
                }))
                continue
            logger.info("tracked arm set to %s", side or "auto")
            await websocket.send_text(json.dumps({"type": "side", "side": side}))
            continue

        if msg.get("type") != "select_motion":
            continue

        motion = msg.get("motion")
        pad = config.MOTION_TO_PAD.get(motion)
        if pad is None:
            await websocket.send_text(json.dumps({
                "type": "status", "pad": None, "state": "error",
                "detail": f"unknown motion '{motion}'",
            }))
            continue

        await websocket.send_text(json.dumps({
            "type": "status", "pad": pad, "state": "firing",
            "duration_ms": config.DEFAULT_DURATION_MS,
        }))
        try:
            await controller.fire(pad)
            await websocket.send_text(json.dumps({"type": "status", "pad": pad, "state": "done"}))
        except ActuationError as exc:
            logger.warning("actuation rejected for pad %s: %s", pad, exc)
            await websocket.send_text(json.dumps({
                "type": "status", "pad": pad, "state": "error", "detail": str(exc),
            }))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue = None
    forward_task = None
    try:
        await runtime.acquire()
        await websocket.send_text(json.dumps({"type": "ready", "armed": controller.armed}))
        queue = broadcaster.subscribe()
        forward_task = asyncio.create_task(_forward_pose(websocket, queue))
        await _handle_client_messages(websocket)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Live WebSocket runtime could not start.")
        with contextlib.suppress(Exception):
            await websocket.send_text(json.dumps({
                "type": "status",
                "pad": None,
                "state": "error",
                "detail": "The live camera or pose model could not be started.",
            }))
            await websocket.close(code=1011)
    finally:
        if queue is not None:
            broadcaster.unsubscribe(queue)
        if forward_task is not None:
            forward_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await forward_task
        await runtime.release()


@app.post("/placement")
async def placement_endpoint(
    relaxed: UploadFile = File(...),
    bicep_flexed: UploadFile = File(...),
    tricep_flexed: UploadFile = File(...),
    front: UploadFile = File(...),
    back: UploadFile = File(...),
):
    """
    5-photo pad-placement calibration. See placement/pipeline.py for the
    exact protocol each file needs to follow. Always returns 200 with a
    per-pad {"ok", "point", "detail", "overlay_b64"} breakdown plus an
    overall "calibration_complete" flag — a partial/failed calibration is
    visible in the response rather than an all-or-nothing error. 422 is
    reserved for malformed input (an upload that isn't a decodable image).
    """
    try:
        return compute_placement(
            relaxed=await relaxed.read(),
            bicep_flexed=await bicep_flexed.read(),
            tricep_flexed=await tricep_flexed.read(),
            front=await front.read(),
            back=await back.read(),
        )
    except PlacementError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
