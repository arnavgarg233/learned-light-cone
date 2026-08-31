"""Minimal ONNX protobuf reader.

Reads pangu_weather_*.onnx (1.18 GB each) over an mmap without copying the
weight bytes. The `onnx` Python package is not installed on this machine and a
full ModelProto parse would materialise 1.1 GB of float arrays that this lane
never needs, so the wire format is walked directly.

Only the fields the reachability compiler needs are decoded:

  ModelProto   ir_version=1, opset_import=8, graph=7
  GraphProto   node=1, name=2, initializer=5, input=11, output=12
  NodeProto    input=1, output=2, name=3, op_type=4, attribute=5
  AttributeProto  name=1, f=2, i=3, s=4, t=5, type=20, floats=7, ints=8,
                  strings=9
  TensorProto  dims=1, data_type=2, float_data=4, int32_data=5, int64_data=7,
               name=8, raw_data=9, double_data=10, uint64_data=11
  ValueInfoProto  name=1, type=2
  TypeProto    tensor_type=1
  Tensor       elem_type=1, shape=2
  TensorShapeProto  dim=1
  Dimension    dim_value=1, dim_param=2

Tensor payloads are recorded as (offset, length) into the mmap. `TensorRef.load`
materialises one on demand, which the compiler only does for the small
shape/index tensors.
"""
from __future__ import annotations

import mmap
import os
import struct

import numpy as np

# ONNX TensorProto.DataType
FLOAT, UINT8, INT8, UINT16, INT16, INT32, INT64 = 1, 2, 3, 4, 5, 6, 7
STRING, BOOL, FLOAT16, DOUBLE, UINT32, UINT64 = 8, 9, 10, 11, 12, 13

_NP = {
    FLOAT: np.float32, UINT8: np.uint8, INT8: np.int8, UINT16: np.uint16,
    INT16: np.int16, INT32: np.int32, INT64: np.int64, BOOL: np.bool_,
    FLOAT16: np.float16, DOUBLE: np.float64, UINT32: np.uint32,
    UINT64: np.uint64,
}


def _varint(buf, i):
    r = 0
    s = 0
    while True:
        b = buf[i]
        i += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, i
        s += 7


def _fields(buf, i, end):
    """Yield (field_number, wire_type, payload_start, payload_end, scalar)."""
    while i < end:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, j = _varint(buf, i)
            yield fn, wt, i, j, v
            i = j
        elif wt == 1:
            yield fn, wt, i, i + 8, None
            i += 8
        elif wt == 2:
            n, j = _varint(buf, i)
            yield fn, wt, j, j + n, None
            i = j + n
        elif wt == 5:
            yield fn, wt, i, i + 4, None
            i += 4
        else:                                          # pragma: no cover
            raise ValueError(f"wire type {wt} at {i}")


def _zigzag(v):
    return (v >> 1) ^ -(v & 1)


class TensorRef:
    """A TensorProto located in the mmap, decoded lazily."""

    __slots__ = ("name", "dims", "data_type", "_buf", "_raw", "_packed")

    def __init__(self, name, dims, data_type, buf, raw, packed):
        self.name = name
        self.dims = tuple(dims)
        self.data_type = data_type
        self._buf = buf
        self._raw = raw            # (start, end) of raw_data, or None
        self._packed = packed      # (field, start, end) of *_data, or None

    @property
    def size(self):
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self):
        if self._raw is not None:
            return self._raw[1] - self._raw[0]
        if self._packed is not None:
            return self._packed[2] - self._packed[1]
        return 0

    def load(self):
        dt = _NP.get(self.data_type)
        if dt is None:
            raise ValueError(f"unsupported dtype {self.data_type} for {self.name}")
        if self._raw is not None:
            a, b = self._raw
            arr = np.frombuffer(self._buf[a:b], dtype=dt)
        elif self._packed is not None:
            f, a, b = self._packed
            if f in (4, 10):                              # float_data, double_data
                w = 4 if f == 4 else 8
                arr = np.frombuffer(self._buf[a:b], dtype=dt if f == 10 else np.float32)
                del w
            else:                                         # varint-packed ints
                vals = []
                i = a
                while i < b:
                    v, i = _varint(self._buf, i)
                    vals.append(v)
                arr = np.array(vals, dtype=np.uint64).astype(np.int64) if f in (7, 11) \
                    else np.array(vals, dtype=np.int64).astype(np.int32)
        else:
            arr = np.zeros(0, dtype=dt)
        if arr.dtype != dt:
            arr = arr.astype(dt)
        if self.dims:
            arr = arr.reshape(self.dims)
        elif arr.size == 1:
            arr = arr.reshape(())
        return arr

    def __repr__(self):
        return f"TensorRef({self.name!r}, {self.dims}, dt={self.data_type})"


class Attr:
    __slots__ = ("name", "i", "f", "s", "ints", "floats", "strings", "t", "type")

    def __init__(self):
        self.name = ""
        self.i = None
        self.f = None
        self.s = None
        self.ints = None
        self.floats = None
        self.strings = None
        self.t = None
        self.type = None

    def key(self):
        """Canonical value used by the deep structural fingerprint."""
        parts = [self.name, str(self.type)]
        if self.i is not None:
            parts.append(f"i={self.i}")
        if self.f is not None:
            parts.append(f"f={self.f!r}")
        if self.s is not None:
            parts.append(f"s={self.s!r}")
        if self.ints is not None:
            parts.append("ints=" + ",".join(map(str, self.ints)))
        if self.floats is not None:
            parts.append("floats=" + ",".join(repr(x) for x in self.floats))
        if self.strings is not None:
            parts.append("strings=" + "|".join(repr(x) for x in self.strings))
        if self.t is not None:
            parts.append(f"t=dims{self.t.dims}:dt{self.t.data_type}")
        return "(" + ";".join(parts) + ")"

    def __repr__(self):
        return f"Attr{self.key()}"


class Node:
    __slots__ = ("op_type", "name", "input", "output", "attr")

    def __init__(self):
        self.op_type = ""
        self.name = ""
        self.input = []
        self.output = []
        self.attr = {}

    def __repr__(self):
        return f"Node({self.op_type}, in={self.input}, out={self.output})"


class Graph:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.buf = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.nodes = []
        self.initializer = {}
        self.inputs = []
        self.outputs = []
        self.ir_version = None
        self.opset = []
        self.graph_name = ""
        self._parse()

    def close(self):
        self.buf.close()
        self.f.close()

    # -- parsing ------------------------------------------------------------
    def _tensor(self, a, b):
        dims, dt, name = [], 0, ""
        raw, packed = None, None
        for fn, wt, s, e, v in _fields(self.buf, a, b):
            if fn == 1 and wt == 0:
                dims.append(v)
            elif fn == 1 and wt == 2:                     # packed dims
                i = s
                while i < e:
                    d, i = _varint(self.buf, i)
                    dims.append(d)
            elif fn == 2:
                dt = v
            elif fn == 8:
                name = self.buf[s:e].decode()
            elif fn == 9:
                raw = (s, e)
            elif fn in (4, 5, 7, 10, 11):
                packed = (fn, s, e)
        return TensorRef(name, dims, dt, self.buf, raw, packed)

    def _attr(self, a, b):
        at = Attr()
        for fn, wt, s, e, v in _fields(self.buf, a, b):
            if fn == 1:
                at.name = self.buf[s:e].decode()
            elif fn == 20:
                at.type = v
            elif fn == 2:
                at.f = struct.unpack("<f", self.buf[s:e])[0]
            elif fn == 3:
                at.i = v if v < (1 << 63) else v - (1 << 64)
            elif fn == 4:
                at.s = self.buf[s:e].decode(errors="replace")
            elif fn == 5:
                at.t = self._tensor(s, e)
            elif fn == 7:
                n = (e - s) // 4
                at.floats = list(struct.unpack(f"<{n}f", self.buf[s:e]))
            elif fn == 8:
                if wt == 2:
                    out, i = [], s
                    while i < e:
                        x, i = _varint(self.buf, i)
                        out.append(x if x < (1 << 63) else x - (1 << 64))
                    at.ints = out
                else:
                    at.ints = (at.ints or []) + [v if v < (1 << 63) else v - (1 << 64)]
            elif fn == 9:
                at.strings = (at.strings or []) + [self.buf[s:e].decode(errors="replace")]
        return at

    def _node(self, a, b):
        nd = Node()
        for fn, wt, s, e, v in _fields(self.buf, a, b):
            if fn == 1:
                nd.input.append(self.buf[s:e].decode())
            elif fn == 2:
                nd.output.append(self.buf[s:e].decode())
            elif fn == 3:
                nd.name = self.buf[s:e].decode()
            elif fn == 4:
                nd.op_type = self.buf[s:e].decode()
            elif fn == 5:
                at = self._attr(s, e)
                nd.attr[at.name] = at
        return nd

    def _value_info(self, a, b):
        name, elem, shape = "", 0, []
        for fn, wt, s, e, v in _fields(self.buf, a, b):
            if fn == 1:
                name = self.buf[s:e].decode()
            elif fn == 2:                                  # TypeProto
                for f2, w2, s2, e2, v2 in _fields(self.buf, s, e):
                    if f2 != 1:                            # tensor_type
                        continue
                    for f3, w3, s3, e3, v3 in _fields(self.buf, s2, e2):
                        if f3 == 1:
                            elem = v3
                        elif f3 == 2:                      # TensorShapeProto
                            for f4, w4, s4, e4, v4 in _fields(self.buf, s3, e3):
                                if f4 != 1:
                                    continue
                                dv, dp = None, None
                                for f5, w5, s5, e5, v5 in _fields(self.buf, s4, e4):
                                    if f5 == 1:
                                        dv = v5
                                    elif f5 == 2:
                                        dp = self.buf[s5:e5].decode()
                                shape.append(dv if dv is not None else dp)
        return {"name": name, "elem_type": elem, "shape": shape}

    def _parse(self):
        n = len(self.buf)
        for fn, wt, s, e, v in _fields(self.buf, 0, n):
            if fn == 1 and wt == 0:
                self.ir_version = v
            elif fn == 8:
                dom, ver = "", None
                for f2, w2, s2, e2, v2 in _fields(self.buf, s, e):
                    if f2 == 1:
                        dom = self.buf[s2:e2].decode()
                    elif f2 == 2:
                        ver = v2
                self.opset.append((dom, ver))
            elif fn == 7:
                self._parse_graph(s, e)

    def _parse_graph(self, a, b):
        for fn, wt, s, e, v in _fields(self.buf, a, b):
            if fn == 1:
                self.nodes.append(self._node(s, e))
            elif fn == 2:
                self.graph_name = self.buf[s:e].decode()
            elif fn == 5:
                t = self._tensor(s, e)
                self.initializer[t.name] = t
            elif fn == 11:
                self.inputs.append(self._value_info(s, e))
            elif fn == 12:
                self.outputs.append(self._value_info(s, e))


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Graph(path)
