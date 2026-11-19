"""Read-only, bounded safetensors metadata/value inspection without Torch or model loading."""

import json
import math
from pathlib import Path

import numpy as np
from safetensors import SafetensorError, safe_open

from .core.checkpoints import file_sha256

MAX_HEADER_BYTES = 16 * 1024 * 1024
DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U64": "<u8",
    "U32": "<u4",
    "U16": "<u2",
    "U8": "u1",
    "BOOL": "?",
    "BF16": "<u2",
}


def decode_values(raw, dtype):
    """Decode a bounded flat prefix; represent nonfinite floats as explicit JSON strings."""
    if dtype not in DTYPES:
        raise ValueError(
            f"Value sampling does not support {dtype}; inspect metadata without --values"
        )
    values = np.frombuffer(raw, dtype=np.dtype(DTYPES[dtype]))
    if dtype == "BF16":
        values = (values.astype(np.uint32) << 16).view(np.float32)
    return [
        v
        if not isinstance(v, float) or math.isfinite(v)
        else ("NaN" if math.isnan(v) else "Infinity" if v > 0 else "-Infinity")
        for v in values.tolist()
    ]


def inspect_safetensors(path, *, tensor=None, limit=100, values=0, checksum=False):
    """Inspect at most 1000 tensor records and 64 prefix values per selected tensor.

    Reading the header is bounded to 16 MiB. Samples use file seeks/short reads, NOT get_tensor
    on an entire multi-GB matrix. --checksum explicitly opts into hashing the whole file.
    """
    path = Path(path)
    if path.suffix != ".safetensors" or not path.is_file():
        raise ValueError("An existing .safetensors file is required; pickle files are never read")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Tensor record limit must be an integer in [1, 1000]")
    if type(values) is not int or not 0 <= values <= 64:
        raise ValueError("Value sample count must be an integer in [0, 64]")
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        header_bytes = int.from_bytes(prefix, "little")
        if not 2 <= header_bytes <= min(MAX_HEADER_BYTES, size - 8):
            raise ValueError("Invalid or oversized safetensors header (maximum 16 MiB)")
        header = json.loads(stream.read(header_bytes))
        try:
            # Let the format implementation validate layouts/offsets/dtypes before raw sampling.
            with safe_open(str(path), framework="np", device="cpu") as tensors:
                keys = sorted(tensors.keys())
                if tensor is not None and tensor not in keys:
                    raise ValueError(f"Tensor not found: {tensor}")
                selected = [tensor] if tensor is not None else keys
                result = []
                for name in selected[:limit]:
                    entry = header[name]
                    record = {
                        "name": name,
                        "dtype": entry["dtype"],
                        "shape": entry["shape"],
                        "numel": math.prod(entry["shape"]),
                        "storage_bytes": entry["data_offsets"][1] - entry["data_offsets"][0],
                    }
                    if values:
                        dtype = entry["dtype"]
                        if dtype not in DTYPES:
                            raise ValueError(
                                f"Value sampling does not support {dtype}; omit --values"
                            )
                        count = min(values, record["numel"])
                        stream.seek(8 + header_bytes + entry["data_offsets"][0])
                        raw = stream.read(count * np.dtype(DTYPES[dtype]).itemsize)
                        if len(raw) != count * np.dtype(DTYPES[dtype]).itemsize:
                            raise ValueError(
                                "Tensor file changed or was truncated during inspection"
                            )
                        record["flat_prefix_values"] = decode_values(raw, dtype)
                    result.append(record)
        except SafetensorError as exc:
            raise ValueError(f"Invalid safetensors file: {exc}") from exc
    output = {
        "file": str(path.resolve()),
        "bytes": size,
        "read_only": True,
        "tensor_count": len(keys),
        "returned": len(result),
        "truncated": len(selected) > len(result),
        "tensors": result,
        "note": "Numeric inspection is not a semantic or causal diagnosis of a model.",
    }
    if checksum:
        output["sha256"] = file_sha256(path)
    return output
