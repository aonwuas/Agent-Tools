
"""
Open WebUI Image Annotation Tools — multimodal retrieval version.

Tools:
  - debug_storage
  - get_image_size
  - crop_image
  - annotate_image
  - grid_overlay
  - retrieve_image

Generated images are saved as Open WebUI files and attached to chat via file events.
retrieve_image returns a standalone image data URI for Open WebUI 0.12.4 multimodal extraction.
Do not wrap retrieval output in Markdown or concatenate it with text.

Security:
  - No arbitrary remote URLs by default.
  - Remote hosts must be explicitly allowlisted.
  - No HTTP redirects.
  - Bounded image bytes, pixels, grids, and annotation inputs.
  - Exact-name filesystem lookup; no recursive scanning.
"""

import asyncio
import base64
import binascii
import io
import ipaddress
import math
import os
import re
import socket
import warnings
import uuid

from pathlib import Path
from typing import List, Optional, Literal
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont, ImageColor, UnidentifiedImageError


UUID_RE = re.compile(
    r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})"
)
HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
MAX_LABEL_LENGTH = 200
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"}
DEFAULT_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_PIXELS = 25_000_000
MAX_GRID_CELLS = 400
MAX_GRID_EDGE = 26


async def _authorized_storage_bytes(fid: str, user: Optional[dict], max_bytes: int):
    """Resolve ID through Open WebUI DB, authorize, then fetch actual storage path.

    Returns (data, diagnosis). Never read a path supplied by the model.
    """
    if not UUID_RE.fullmatch(fid):
        return None, "invalid file ID"
    if not isinstance(user, dict) or not user.get("id"):
        return None, "user context unavailable"
    try:
        from open_webui.models.files import Files
        from open_webui.storage.provider import Storage
        file = await Files.get_file_by_id(fid)
        if file is None:
            return None, "file ID not present in Open WebUI database"

        user_id = str(user["id"])
        if str(file.user_id) != user_id:
            # Never trust a caller-supplied role alone; get the server-side user.
            from open_webui.models.users import Users
            actual_user = await Users.get_user_by_id(user_id)
            if actual_user is None:
                return None, "user account not found"
            if getattr(actual_user, "role", None) != "admin":
                from open_webui.utils.access_control.files import has_access_to_file
                if not await has_access_to_file(fid, "read", actual_user):
                    return None, "file access denied"

        if not file.path:
            return None, "file has no storage path"
        storage_path = await asyncio.to_thread(Storage.get_file, file.path)
        if not isinstance(storage_path, (str, os.PathLike)):
            return None, "storage provider did not return a path"
        path = Path(storage_path)
        if not path.is_file():
            return None, "storage provider path does not exist"
        if path.stat().st_size > max_bytes:
            return None, "image exceeds configured byte limit"
        data = await asyncio.to_thread(path.read_bytes)
        if not data or len(data) > max_bytes:
            return None, "image empty or exceeds configured byte limit"
        return data, "readable via Open WebUI storage provider"
    except Exception as exc:
        return None, f"Open WebUI storage error ({type(exc).__name__})"


class Annotation(BaseModel):
    kind: Literal["arrow", "box", "ellipse", "line", "point", "text"] = Field(
        ..., description="Required: arrow, box, ellipse, line, point, or text."
    )
    coords: List[float] = Field(
        ...,
        min_length=2,
        max_length=4,
        description=(
            "Normalized coordinates 0..1. Point/text: [x,y]. "
            "Box/ellipse/arrow/line: [x0,y0,x1,y1]. Values must be between 0 and 1."
        ),
    )
    label: Optional[str] = Field(
        None, max_length=MAX_LABEL_LENGTH,
        description="Optional short label."
    )
    color: str = Field("#ff2d20", description="Color: preferred #RRGGBB hex, e.g. #FF0000 red, #00FF00 green, #0000FF blue. Common names (red, cyan, blue) are also accepted. Omit for default red.")
    width: Optional[int] = Field(
        None, ge=1, le=64, description="Stroke width in pixels."
    )


class Tools:
    class Valves(BaseModel):
        max_edge: int = Field(3072, ge=128, le=4096)
        min_crop_edge: int = Field(512, ge=64, le=2048)
        max_image_bytes: int = Field(DEFAULT_MAX_BYTES, ge=1024)
        max_image_pixels: int = Field(DEFAULT_MAX_PIXELS, ge=1024)
        allow_remote_urls: bool = Field(
            False,
            description="Enable remote image URLs only when needed."
        )
        allowed_remote_hosts: str = Field(
            "",
            description=(
                "Comma-separated exact hostnames allowed for remote images. "
                "Remote fetching remains disabled unless enabled separately."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()
        self._session = requests.Session()

    # ---- validation and image loading -------------------------------------

    @staticmethod
    def _is_image(file_info: dict) -> bool:
        if not isinstance(file_info, dict):
            return False
        kind = str(file_info.get("type", "")).lower()
        content_type = str(file_info.get("content_type", "")).lower()
        mime = str(file_info.get("mime_type", "")).lower()
        return (
            kind.startswith("image")
            or content_type.startswith("image/")
            or mime.startswith("image/")
        )

    def _decode_image(self, data: bytes) -> Image.Image:
        """Decode a bounded image, rejecting unsupported formats and huge images."""
        if not data or len(data) > self.valves.max_image_bytes:
            raise ValueError("Image is empty or exceeds the configured byte limit.")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    if source.format not in ALLOWED_FORMATS:
                        raise ValueError("Unsupported image format.")
                    width, height = source.size
                    if width < 1 or height < 1:
                        raise ValueError("Invalid image dimensions.")
                    if width * height > self.valves.max_image_pixels:
                        raise ValueError("Image exceeds the configured pixel limit.")
                    if getattr(source, "n_frames", 1) > 1:
                        # Annotation tools operate on a still image.
                        source.seek(0)
                    source.load()
                    return source.convert("RGBA")

        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ValueError("Image rejected by decompression-bomb protection.")
        except UnidentifiedImageError:
            raise ValueError("The file is not a recognized image.")
        except OSError as exc:
            raise ValueError("Image could not be decoded.") from exc

    @staticmethod
    def _extract_id(ref: str) -> Optional[str]:
        match = UUID_RE.search(ref or "")
        return match.group(1) if match else None

    def _remote_host_allowed(self, url: str) -> bool:
        """Allow only explicitly listed public hosts, with no redirects."""
        if not self.valves.allow_remote_urls:
            return False

        try:
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.hostname:
                return False
            if parsed.username or parsed.password:
                return False

            allowed = {
                host.strip().lower().rstrip(".")
                for host in self.valves.allowed_remote_hosts.split(",")
                if host.strip()
            }
            host = parsed.hostname.lower().rstrip(".")
            if host not in allowed:
                return False

            # Reject private, loopback, link-local, reserved, or local IPs.
            try:
                addresses = {ipaddress.ip_address(host)}
            except ValueError:
                try:
                    addresses = {
                        ipaddress.ip_address(item[4][0])
                        for item in socket.getaddrinfo(
                            host, parsed.port or 443, type=socket.SOCK_STREAM
                        )
                    }
                except (OSError, ValueError):
                    return False

            return bool(addresses) and all(
                address.is_global for address in addresses
            )

        except (ValueError, TypeError):
            return False

    def _fetch(
        self,
        url: str,
        token: Optional[str],
        tried: list,
        timeout: int = 5,
        remote: bool = False,
    ) -> Optional[bytes]:
        """Bounded HTTP read. Redirects are deliberately disabled."""
        if remote and not self._remote_host_allowed(url):
            tried.append("remote URL blocked by host policy")
            return None

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            with self._session.get(
                url,
                headers=headers,
                timeout=(min(timeout, 5), timeout),
                stream=True,
                allow_redirects=False,
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    tried.append("HTTP redirect blocked")
                    return None

                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if content_type and not content_type.lower().startswith("image/"):
                    tried.append("HTTP response was not an image MIME type")
                    return None

                length = response.headers.get("Content-Length")
                if length:
                    try:
                        if int(length) > self.valves.max_image_bytes:
                            tried.append("HTTP response exceeds byte limit")
                            return None
                    except ValueError:
                        pass

                chunks, total = [], 0
                for chunk in response.iter_content(64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > self.valves.max_image_bytes:
                        tried.append("HTTP response exceeds byte limit")
                        return None
                    chunks.append(chunk)

                data = b"".join(chunks)
                if not data:
                    tried.append("HTTP response was empty")
                    return None
                return data

        except requests.RequestException as exc:
            tried.append(f"HTTP fetch failed ({type(exc).__name__})")
        except Exception as exc:
            tried.append(f"HTTP processing failed ({type(exc).__name__})")
        return None

    @staticmethod
    def _collect_refs(ref, files, messages):
        """Explicit ref first; then current attachments; then recent user images."""
        refs = []
        if isinstance(ref, str) and ref.strip():
            refs.append(ref.strip())
        def add(item):
            if not isinstance(item, dict):
                return
            category = str(item.get("type", "")).lower()
            if category not in ("", "file", "image", "image/png", "image/jpeg", "image/webp") and not category.startswith("image/"):
                return
            mime = str(item.get("content_type") or item.get("mime_type") or "").lower()
            if mime and not mime.startswith("image/"):
                return
            for field in ("id", "url", "path"):
                value = item.get(field)
                if isinstance(value, str) and value.strip():
                    refs.append(value.strip())
                    break
            nested = item.get("file")
            if isinstance(nested, dict):
                add(nested)
        for item in files or []:
            add(item)
        if not refs and isinstance(messages, list):
            for msg in reversed(messages[-50:]):
                if not isinstance(msg, dict) or msg.get("role") != "user":
                    continue
                for item in msg.get("files", []) or []:
                    add(item)
                for item in msg.get("attachments", []) or []:
                    add(item)
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                            target = part.get("image_url") or part.get("url")
                            if isinstance(target, dict):
                                target = target.get("url")
                            if isinstance(target, str):
                                refs.append(target)
                if refs:
                    break
        return list(dict.fromkeys(refs))

    async def _load(self, ref, files, user, messages=None):
        refs = self._collect_refs(ref, files, messages)
        if not refs:
            raise ValueError("No image reference found. Supply its Open WebUI file ID.")
        tried = []
        for candidate in refs:
            if candidate.startswith("data:"):
                try:
                    header, encoded = candidate.split(",", 1)
                    if not header.lower().startswith("data:image/") or ";base64" not in header.lower():
                        raise ValueError("Only image base64 data URIs are supported")
                    if len(encoded) > self.valves.max_image_bytes * 4 // 3 + 16:
                        raise ValueError("Data URI exceeds byte limit")
                    return self._decode_image(base64.b64decode(encoded, validate=True))
                except (ValueError, binascii.Error) as exc:
                    tried.append(f"data URI rejected ({type(exc).__name__})")
                continue

            parsed = urlparse(candidate)
            if parsed.scheme.lower() in ("http", "https"):
                # Only explicitly allowed public remote hosts. Internal API
                # URLs are never fetched; their UUID can be resolved locally.
                if self._remote_host_allowed(candidate):
                    data = await asyncio.to_thread(
                        self._fetch, candidate, None, tried, 5, True
                    )
                    if data:
                        try:
                            return self._decode_image(data)
                        except ValueError:
                            tried.append("remote image validation failed")
                else:
                    fid = self._extract_id(parsed.path) if "/api/files/" in parsed.path else None
                    if fid:
                        data, status = await _authorized_storage_bytes(
                            fid, user, self.valves.max_image_bytes
                        )
                        if data is not None:
                            return self._decode_image(data)
                        tried.append(status)
                    else:
                        tried.append("remote URL blocked or not allowlisted")
                continue

            fid = self._extract_id(candidate)
            if not fid:
                tried.append("reference is not a recognized image ID")
                continue
            data, status = await _authorized_storage_bytes(fid, user, self.valves.max_image_bytes)
            if data is not None:
                return self._decode_image(data)
            tried.append(status)

        raise ValueError("Could not load a valid image. " + "; ".join(tried[-5:]))

    # ---- output helpers ----------------------------------------------------

    async def _publish_image(self, img: Image.Image, name: str,
                             user: dict, emitter=None, jpeg=True) -> str:
        """Persist an image in the installed Open WebUI 0.11.3 file system.

        The tool result stays short; the chat UI receives a file attachment.
        """
        if not isinstance(user, dict) or not user.get("id"):
            raise ValueError("Open WebUI user context missing; cannot save output image.")
        if max(img.size) > self.valves.max_edge:
            img = img.copy()
            img.thumbnail((self.valves.max_edge, self.valves.max_edge),
                          Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        if jpeg:
            img.convert("RGB").save(buf, "JPEG", quality=82, optimize=True)
            mime, ext = "image/jpeg", "jpg"
        else:
            img.save(buf, "PNG", optimize=True)
            mime, ext = "image/png", "png"
        data = buf.getvalue()
        if len(data) > self.valves.max_image_bytes:
            raise ValueError("Generated image exceeds byte limit.")
        from open_webui.storage.provider import Storage
        from open_webui.models.files import Files, FileForm
        fid = str(uuid.uuid4())
        filename = f"{fid}_{name}.{ext}"
        contents, file_path = await asyncio.to_thread(
            Storage.upload_file, io.BytesIO(data), filename, {}
        )
        try:
            file = await Files.insert_new_file(
                str(user["id"]),
                FileForm(id=fid, filename=f"{name}.{ext}", path=file_path,
                         data={}, meta={"name": f"{name}.{ext}",
                                        "content_type": mime,
                                        "size": len(contents)}),
            )
            if file is None:
                raise RuntimeError("File database insertion failed")
        except Exception:
            await asyncio.to_thread(Storage.delete_file, file_path)
            raise
        url = f"/api/v1/files/{fid}/content"
        attached = {"type": "image", "id": fid, "url": url,
                    "name": f"{name}.{ext}", "content_type": mime}
        if emitter is not None:
            await emitter({"type": "chat:message:files",
                           "data": {"files": [attached]}})
            shown = "Image attached to the chat."
        else:
            shown = "No chat event emitter was provided; use the file URL to view it."
        return (f"Created {name}: {img.width}x{img.height} px; "
                f"image_ref={fid}; url={url}. {shown} "
                "This is a saved image file, not image pixels in the model's tool text. "
                "Do not claim you visually inspected the result unless you actually received it as an image.")

    @staticmethod
    def _font(px: int):
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ):
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, px)
                except OSError:
                    pass
        return ImageFont.load_default()

    @staticmethod
    def _validate_coords(kind: str, coords: list) -> None:
        expected = 2 if kind in ("point", "text") else 4
        if len(coords) != expected:
            raise ValueError(
                f"{kind} requires exactly {expected} coordinates."
            )
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords):
            raise ValueError("Coordinates must be finite values between 0 and 1.")

    @staticmethod
    def _px(box, width, height):
        x0, y0, x1, y1 = box
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return (
            max(0, min(width, round(x0 * width))),
            max(0, min(height, round(y0 * height))),
            max(0, min(width, round(x1 * width))),
            max(0, min(height, round(y1 * height))),
        )

    @staticmethod
    def _color(value: str):
        if not isinstance(value, str):
            raise ValueError("Color must be an RGB color string, such as #FF0000 or red.")
        try:
            rgb = ImageColor.getrgb(value.strip())
            if len(rgb) != 3:
                raise ValueError("alpha channels are not supported")
            return "#{:02X}{:02X}{:02X}".format(*rgb)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Unsupported color {value!r}. Use #RRGGBB or a basic color name such as red, blue, or cyan.") from exc

    @staticmethod
    def _metadata_files(metadata: Optional[dict]) -> list:
        if not isinstance(metadata, dict):
            return []
        files = metadata.get("files", [])
        return files if isinstance(files, list) else []

    # ---- diagnostics -------------------------------------------------------

    async def debug_storage(
        self,
        ref: str = Field("", description="Optional file ID or URL."),
        __metadata__: dict = None,
        __user__: dict = None,
        __files__: list = None,
        __messages__: list = None,
    ) -> str:
        """Report storage resolution without exposing paths, tokens or private metadata."""
        files = (__files__ or []) + self._metadata_files(__metadata__)
        refs = self._collect_refs(ref, files, __messages__)
        lines = [
            f"current attached files: {len(files)}",
            f"image reference candidates: {len(refs)}",
            f"user context available: {bool(isinstance(__user__, dict) and __user__.get('id'))}",
        ]
        if not refs:
            return "\n".join(lines + ["No image reference available."])
        fid = self._extract_id(refs[0])
        if fid:
            data, status = await _authorized_storage_bytes(fid, __user__, self.valves.max_image_bytes)
            lines.append("Open WebUI storage: " + status)
        try:
            img = await self._load(ref, files, __user__, __messages__)
            lines.append(f"Image load: OK ({img.width}x{img.height})")
        except Exception as exc:
            lines.append(f"Image load: failed ({type(exc).__name__}): {exc}")
        return "\n".join(lines)

    # ---- tools -------------------------------------------------------------

    async def get_image_size(
        self,
        ref: str = Field("", description="Optional file ID, data URI, or allowed URL."),
        __metadata__: dict = None,
        __user__: dict = None,
        __files__: list = None,
        __messages__: list = None,
    ) -> str:
        """Return the pixel dimensions of the target image."""
        try:
            image = await self._load(
                ref, (__files__ or []) + self._metadata_files(__metadata__),
                __user__, __messages__
            )
            return f"Image is {image.width}x{image.height} px."
        except Exception as exc:
            return f"Error: {exc}"

    async def crop_image(
        self,
        x0: float = Field(..., ge=0, le=1),
        y0: float = Field(..., ge=0, le=1),
        x1: float = Field(..., ge=0, le=1),
        y1: float = Field(..., ge=0, le=1),
        ref: str = Field("", description="Optional image reference."),
        __metadata__: dict = None,
        __user__: dict = None,
        __files__: list = None,
        __messages__: list = None,
        __event_emitter__=None,
    ) -> str:
        """Crop and enlarge an image region; coordinates are normalized 0..1."""
        try:
            image = await self._load(
                ref, (__files__ or []) + self._metadata_files(__metadata__),
                __user__, __messages__
            )
            width, height = image.size
            box = self._px((x0, y0, x1, y1), width, height)

            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                return f"Error: crop too small: {box} of {width}x{height}."

            crop = image.crop(box)
            longest = max(crop.size)
            if longest < self.valves.min_crop_edge:
                scale = self.valves.min_crop_edge / longest
                crop = crop.resize(
                    (
                        max(1, round(crop.width * scale)),
                        max(1, round(crop.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )

            context = image.copy()
            context.thumbnail((320, 320), Image.Resampling.LANCZOS)
            sx, sy = context.width / width, context.height / height
            draw = ImageDraw.Draw(context)
            draw.rectangle(
                [
                    box[0] * sx, box[1] * sy,
                    box[2] * sx, box[3] * sy,
                ],
                outline=(255, 40, 40, 255),
                width=3,
            )

            # Publish a single composite so the chat-file event does not overwrite
            # another image attachment within the same tool call.
            preview_width = crop.width + context.width + 16
            preview_height = max(crop.height, context.height)
            preview = Image.new("RGB", (preview_width, preview_height), "#202020")
            preview.paste(crop.convert("RGB"), (0, 0))
            preview.paste(context.convert("RGB"), (crop.width + 16, 0))
            response = await self._publish_image(preview, "crop", __user__, __event_emitter__)
            return (response + f" Original region: pixel box {box} of {width}x{height}; "
                    "left pane is enlarged crop, right pane shows location. "
                    "Upscaling does not restore missing details.")
        except Exception as exc:
            return f"Error: {exc}"

    # ---- annotation presentation ------------------------------------------

    @staticmethod
    def _anchor(item):
        c = item.coords
        if len(c) == 2:
            return (c[0], c[1])
        if item.kind in ("box", "ellipse"):
            return ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2)
        return (c[2], c[3])

    @staticmethod
    def _layout_mode(items, requested):
        if requested != "auto":
            return requested
        n = len(items)
        if n <= 4:
            anchors = [Tools._anchor(i) for i in items]
            close = sum(1 for k, a in enumerate(anchors) for b in anchors[k+1:]
                        if (a[0]-b[0])**2 + (a[1]-b[1])**2 < .015)
            if not close:
                return "direct"
        return "numbered"

    @staticmethod
    def _clamp_text(draw, xy, label, font, width, height, pad=5):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x = max(pad, min(int(xy[0]), width-tw-2*pad))
        y = max(pad, min(int(xy[1]), height-th-2*pad))
        return x, y, tw, th, bbox

    def _render_marks(self, image, items, numbered=False):
        width, height = image.size
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        base = max(2, min(24, round(min(width, height) * .004)))
        font = self._font(max(12, min(32, round(min(width,height) * .024))))
        occupied = []
        for index, item in enumerate(items, 1):
            color = self._color(item.color)
            stroke = item.width or base
            c = item.coords
            xy = [(c[i] * (width if i%2==0 else height)) for i in range(len(c))]
            ax, ay = self._anchor(item)
            px, py = ax*width, ay*height
            if item.kind == "arrow":
                x0,y0,x1,y1=xy
                draw.line((x0,y0,x1,y1), fill=color, width=stroke)
                angle=math.atan2(y1-y0,x1-x0)
                length=max(10,stroke*4)
                for delta in (-.45,.45):
                    draw.line([(x1,y1),(x1-length*math.cos(angle+delta),y1-length*math.sin(angle+delta))],fill=color,width=stroke)
            elif item.kind in ("box","ellipse"):
                rect=self._px(c,width,height)
                if item.kind=="box":
                    draw.rectangle(rect,outline=color,width=stroke)
                else:
                    draw.ellipse(rect,outline=color,width=stroke)
            elif item.kind=="line":
                draw.line(tuple(xy),fill=color,width=stroke)
            elif item.kind=="point":
                r=max(5,stroke*2)
                draw.ellipse((px-r,py-r,px+r,py+r),outline=color,width=stroke)
            # Marker is anchored at the target even for directional arrows.
            if numbered:
                r=max(12,min(20,round(min(width,height)*.017)))
                draw.ellipse((px-r,py-r,px+r,py+r),fill=(20,24,30,240),outline=color,width=3)
                value=str(index)
                bb=draw.textbbox((0,0),value,font=font)
                draw.text((px-(bb[2]-bb[0])/2-bb[0],py-(bb[3]-bb[1])/2-bb[1]),value,font=font,fill="white")
            elif item.label:
                # Choose the first collision-free nearby label location.
                candidates=[(px+8,py-28),(px+8,py+12),(px-160,py-28),
                            (px-160,py+12),(px+8,py-55),(px-160,py-55)]
                best=None; lowest=float('inf')
                for candidate in candidates:
                    x,y,tw,th,bb=self._clamp_text(draw,candidate,item.label,font,width,height)
                    rect=(x-5,y-5,x+tw+9,y+th+9)
                    overlap=sum(max(0,min(rect[2],old[2])-max(rect[0],old[0]))*
                                max(0,min(rect[3],old[3])-max(rect[1],old[1])) for old in occupied)
                    if overlap < lowest:
                        best=(x,y,tw,th,bb,rect); lowest=overlap
                    if not overlap: break
                x,y,tw,th,bb,rect=best
                occupied.append(rect)
                draw.rounded_rectangle(rect,radius=5,fill=(12,12,12,225),outline=color,width=2)
                draw.text((x-bb[0],y-bb[1]),item.label,fill="white",font=font)
        return Image.alpha_composite(image.convert("RGBA"),layer)

    def _render_legend(self, image, items, title="Annotation key"):
        w,h=image.size
        font=self._font(19)
        small=self._font(17)
        # A separate right-hand key keeps the original image unobscured.
        key_w=max(280,min(500,round(w*.45)))
        row_h=29
        key_h=max(h,56+len(items)*row_h)
        canvas=Image.new("RGB",(w+key_w,key_h),(245,247,250))
        canvas.paste(image.convert("RGB"),(0,0))
        draw=ImageDraw.Draw(canvas)
        draw.text((w+15,14),title,fill=(20,26,35),font=font)
        for j,item in enumerate(items):
            y=50+j*row_h
            color=self._color(item.color)
            draw.ellipse((w+14,y+3,w+35,y+24),fill=color)
            draw.text((w+24,y+14),str(j+1),anchor="mm",font=small,fill="#101010")
            # Wrap long labels without clipping into the image area.
            label=(item.label or item.kind).replace("\n"," ")
            max_chars=max(18,int((key_w-70)/9.0))
            if len(label)>max_chars:
                label=label[:max_chars-1]+"…"
            draw.text((w+46,y+3),label,fill=(30,34,42),font=small)
        return canvas

    @staticmethod
    def _spatial_groups(items, max_groups=4):
        # Deterministic recursive spatial partition, independent of subject type.
        groups=[list(enumerate(items))]
        while len(groups)<max_groups:
            options=[(len(g),idx) for idx,g in enumerate(groups) if len(g)>=6]
            if not options: break
            _,idx=max(options)
            group=groups.pop(idx)
            anchors=[Tools._anchor(item) for _,item in group]
            xs=[p[0] for p in anchors]; ys=[p[1] for p in anchors]
            axis=0 if max(xs)-min(xs)>=max(ys)-min(ys) else 1
            group.sort(key=lambda pair: Tools._anchor(pair[1])[axis])
            mid=len(group)//2
            groups[idx:idx]=[group[:mid],group[mid:]]
        return groups

    def _render_regional(self, image, items, detail=False):
        import copy
        groups=self._spatial_groups(items, max_groups=4)
        w,h=image.size
        panes=[]
        for number,group in enumerate(groups,1):
            anchors=[self._anchor(item) for _,item in group]
            minx,maxx=min(x for x,y in anchors),max(x for x,y in anchors)
            miny,maxy=min(y for x,y in anchors),max(y for x,y in anchors)
            # Account for entire boxes and lines as well as annotation anchors.
            for _,item in group:
                c=item.coords
                minx=min(minx,*c[0::2]);maxx=max(maxx,*c[0::2])
                miny=min(miny,*c[1::2]);maxy=max(maxy,*c[1::2])
            pad=.09
            x0=max(0,minx-pad);y0=max(0,miny-pad)
            x1=min(1,maxx+pad);y1=min(1,maxy+pad)
            # Avoid tiny crops for features close together.
            if x1-x0 < .20:
                extra=(.20-(x1-x0))/2;x0=max(0,x0-extra);x1=min(1,x1+extra)
            if y1-y0 < .20:
                extra=(.20-(y1-y0))/2;y0=max(0,y0-extra);y1=min(1,y1+extra)
            box=(int(x0*w),int(y0*h),max(int(x0*w)+1,int(x1*w)),max(int(y0*h)+1,int(y1*h)))
            crop=image.crop(box)
            remapped=[]
            for _,item in group:
                replica=item.model_copy(deep=True)
                replica.coords=[max(0,min(1,(v-(x0 if i%2==0 else y0))/((x1-x0) if i%2==0 else (y1-y0)))) for i,v in enumerate(item.coords)]
                remapped.append(replica)
            rendered=self._render_marks(crop,remapped,numbered=True)
            panel=self._render_legend(rendered,remapped,title=f"Region {number} key")
            panel.thumbnail((1050,740),Image.Resampling.LANCZOS)
            panes.append(panel)
        # Contact sheet delivers a single view and one chat attachment.
        total_w=max(p.width for p in panes)
        header=0
        overview=None
        if detail:
            overview=self._render_marks(image,items,numbered=True)
            overview.thumbnail((total_w,740),Image.Resampling.LANCZOS)
            header=overview.height+35
        total_h=header+sum(p.height+30 for p in panes)+20
        result=Image.new("RGB",(total_w,total_h),(232,236,242))
        draw=ImageDraw.Draw(result)
        y=0
        if overview is not None:
            result.paste(overview,(0,0));y=overview.height+35
        for i,panel in enumerate(panes,1):
            draw.text((10,y-25),f"Region {i}",fill="#1a2530",font=self._font(18))
            result.paste(panel,(0,y));y+=panel.height+30
        return result

    async def retrieve_image(
        self,
        ref: str = Field(
            ...,
            description=(
                "Open WebUI file ID for an existing original or generated image. "
                "Use the image_ref returned by annotate_image, grid_overlay or crop_image. "
                "Call when you need to SEE a previously generated image. "
                "This returns image pixels to the vision model, not merely a URL. "
                "Do not ask the user to reattach a saved image when its ID is available."
            ),
        ),
        max_edge: int = Field(
            1536, ge=256, le=3072,
            description="Maximum width or height of returned image in pixels. "
                        "Default 1536 balances legibility and vision-token cost; "
                        "use a larger value only when needed for fine details."
        ),
        __user__: dict = None,
    ) -> str:
        """Retrieve a saved image by ID and SHOW it to the vision model.

        Use after annotate_image/grid_overlay/crop_image when a generated image
        must be visually inspected. Give the returned image_ref as ref. The
        result is a SINGLE standalone image data URI; Open WebUI 0.12.4 extracts
        image data from tool results and routes it as multimodal content rather
        than text in compatible Chat Completions flows. Do not call repeatedly
        if the image has already been received visually.

        Do not replace this result with a Markdown image, status text, or JSON:
        the native tool image extractor requires a standalone data URI.
        """
        try:
            fid = self._extract_id(ref)
            if fid is None:
                return "Error: Provide a valid Open WebUI image ID as ref."
            # Verify through database-backed storage and file authorization.
            data, diagnostic = await _authorized_storage_bytes(
                fid, __user__, self.valves.max_image_bytes
            )
            if data is None:
                return f"Error: Could not retrieve image: {diagnostic}"
            image = self._decode_image(data)
            if max(image.size) > max_edge:
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            # JPEG is much smaller than PNG for typical annotated images.
            image.convert("RGB").save(buffer, format="JPEG", quality=82, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            # CRITICAL: data URI must be entire return value, without Markdown
            # wrappers or explanatory text, for native Open WebUI image extraction.
            return "data:image/jpeg;base64," + encoded
        except Exception as exc:
            return f"Error: Image retrieval failed ({type(exc).__name__}): {exc}"

    async def annotate_image(
        self,
        annotations: List[dict] = Field(
            ..., min_length=1, max_length=100,
            description=("JSON ARRAY of annotation objects (not a string). Example: "
                         "[{\"kind\":\"point\",\"coords\":[0.5,0.5],\"label\":\"Center\",\"color\":\"red\"}]. "
                         "Kinds: arrow, box, ellipse, line, point, text. "
                         "Coordinates normalized 0..1. point/text need [x,y], others [x0,y0,x1,y1]. "
                         "Colors support #RRGGBB and names such as red/blue/cyan. "
                         "Keep labels short; leave numbering, legends, and layout to the tool."),
        ),
        ref: str = Field("", description="Open WebUI image file ID; omit for a recent attached image."),
        mode: Literal["auto","direct","numbered","sectioned","detail_panels"] = Field(
            "auto", description=("Visual presentation mode. auto (recommended): direct for a few isolated marks, "
                                 "otherwise numbered markers with a legend; "
                                 "direct: full labels positioned around features; "
                                 "numbered: unobtrusive numbered markers plus external legend; "
                                 "sectioned: one contact sheet of spatially grouped annotated crops; "
                                 "detail_panels: overview and enlarged regional crops in one sheet. "
                                 "Choose explicit mode only when user asks or close-up detail is essential.")),
        __metadata__: dict = None,
        __user__: dict = None,
        __files__: list = None,
        __messages__: list = None,
        __event_emitter__=None,
    ) -> str:
        """Annotate an image with readable automatic layouts, then attach it in chat.

        Default mode='auto'. Do NOT assign numbers yourself. Provide only feature
        kind, coordinates, label, and optionally color/width. Supported modes:
        auto, direct, numbered, sectioned, detail_panels. Auto generally uses
        direct labels for <=4 isolated features, numbered with legend otherwise.
        Use sectioned for separated dense clusters and detail_panels when
        magnification plus an overview would help. Tool handles all layout.
        """
        try:
            if not isinstance(annotations,list) or not 1<=len(annotations)<=100:
                raise ValueError("annotations must be a JSON array of 1–100 objects")
            items=[]
            for idx, raw in enumerate(annotations,1):
                try:
                    item=raw if isinstance(raw,Annotation) else Annotation.model_validate(raw)
                    self._validate_coords(item.kind,item.coords)
                    self._color(item.color)
                    items.append(item)
                except Exception as exc:
                    raise ValueError(f"Annotation {idx} is invalid: {exc}") from exc
            image=await self._load(ref,(__files__ or [])+self._metadata_files(__metadata__),__user__,__messages__)
            selected=self._layout_mode(items,mode)
            if selected=="direct":
                rendered=self._render_marks(image,items,numbered=False)
            elif selected=="numbered":
                rendered=self._render_legend(self._render_marks(image,items,numbered=True),items)
            else:
                rendered=self._render_regional(image,items,detail=(selected=="detail_panels"))
            response=await self._publish_image(rendered,f"annotated_{selected}",__user__,__event_emitter__)
            return f"Mode: {selected}; {len(items)} annotations. {response}"
        except Exception as exc:
            return f"Error: {exc}"

    async def grid_overlay(
        self,
        cols: int = Field(10, ge=1, le=MAX_GRID_EDGE),
        rows: int = Field(10, ge=1, le=MAX_GRID_EDGE),
        ref: str = Field("", description="Optional image reference."),
        __metadata__: dict = None,
        __user__: dict = None,
        __files__: list = None,
        __messages__: list = None,
        __event_emitter__=None,
    ) -> str:
        """Overlay a labeled grid; columns A-Z and rows 1-26."""
        try:
            if cols * rows > MAX_GRID_CELLS:
                return f"Error: grid cannot exceed {MAX_GRID_CELLS} cells."

            image = await self._load(
                ref, (__files__ or []) + self._metadata_files(__metadata__),
                __user__, __messages__
            )
            width, height = image.size
            layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            font_size = max(12, min(36, round(min(width, height) * 0.03)))
            font = self._font(font_size)

            for i in range(1, cols):
                x = round(width * i / cols)
                draw.line(
                    [(x, 0), (x, height)],
                    fill=(255, 255, 255, 170), width=2,
                )

            for j in range(1, rows):
                y = round(height * j / rows)
                draw.line(
                    [(0, y), (width, y)],
                    fill=(255, 255, 255, 170), width=2,
                )

            for i in range(cols):
                for j in range(rows):
                    label = f"{chr(65 + i)}{j + 1}"
                    center_x = width * (i + 0.5) / cols
                    center_y = height * (j + 0.5) / rows
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    x = center_x - tw / 2 - bbox[0]
                    y = center_y - th / 2 - bbox[1]
                    pad = 3

                    draw.rectangle(
                        [
                            x - pad, y - pad,
                            x + tw + pad, y + th + pad,
                        ],
                        fill=(0, 0, 0, 175),
                    )
                    draw.text(
                        (x, y), label,
                        fill=(255, 255, 255, 240), font=font,
                    )

            return await self._publish_image(
                Image.alpha_composite(image, layer), "grid", __user__, __event_emitter__
            )
        except Exception as exc:
            return f"Error: {exc}"
