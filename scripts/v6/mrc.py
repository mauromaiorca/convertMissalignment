from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path


class MrcValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MrcHeader:
    path: str
    nx: int
    ny: int
    nz: int
    mode: int
    pixel_size_A: float
    file_size_bytes: int
    extended_header_bytes: int

    @property
    def shape_xyz(self) -> list[int]:
        return [self.nx, self.ny, self.nz]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["shape_xyz"] = self.shape_xyz
        return data


def read_header(path: Path) -> MrcHeader:
    p = Path(path)
    if not p.is_file():
        raise MrcValidationError(f"MRC file not found: {p}")
    size = p.stat().st_size
    if size < 1024:
        raise MrcValidationError(f"MRC file is too small to contain a header: {p}")
    with p.open("rb") as fh:
        header = fh.read(1024)
    nx, ny, nz, mode = struct.unpack_from("<4i", header, 0)
    mx, my, mz = struct.unpack_from("<3i", header, 28)
    cella_x, cella_y, _cella_z = struct.unpack_from("<3f", header, 40)
    nsymbt = struct.unpack_from("<i", header, 92)[0]
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise MrcValidationError(f"invalid MRC dimensions for {p}: {nx}x{ny}x{nz}")
    if mode not in (0, 1, 2, 6):
        raise MrcValidationError(f"unsupported MRC mode for {p}: {mode}")
    pixel = 0.0
    if mx > 0 and math.isfinite(cella_x) and cella_x > 0:
        pixel = float(cella_x) / float(mx)
    elif my > 0 and math.isfinite(cella_y) and cella_y > 0:
        pixel = float(cella_y) / float(my)
    if not math.isfinite(pixel) or pixel <= 0:
        raise MrcValidationError(f"invalid/non-positive MRC pixel size for {p}: {pixel}")
    if nsymbt < 0:
        raise MrcValidationError(f"invalid negative MRC extended-header size for {p}: {nsymbt}")
    bytes_per_pixel = {0: 1, 1: 2, 2: 4, 6: 2}[mode]
    expected_min = 1024 + nsymbt + nx * ny * nz * bytes_per_pixel
    if size < expected_min:
        raise MrcValidationError(
            f"MRC data are truncated for {p}: size={size}, expected at least {expected_min}"
        )
    return MrcHeader(str(p.resolve()), nx, ny, nz, mode, pixel, size, nsymbt)


def validate_stack(path: Path, *, expected_tilts: int | None = None) -> MrcHeader:
    header = read_header(path)
    if expected_tilts is not None and header.nz != expected_tilts:
        raise MrcValidationError(
            f"tilt section mismatch for {path}: stack nz={header.nz}, tilt count={expected_tilts}"
        )
    return header

