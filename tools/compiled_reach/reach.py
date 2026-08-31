"""Static reachability over an ONNX graph. No inference.

Every tensor in the graph is given two facts:

  `const`  the concrete value, when it is derivable from graph constants,
           initializer values and `Shape` of already known shapes, and is small
           enough to hold. A tensor with a concrete value cannot depend on the
           perturbed input column, so its dependency is empty by construction.
  `dep`    a boolean array over the tensor's logical shape, stored
           broadcast-compressed: each axis of the stored array is either 1,
           meaning the whole axis carries one common value, or the full logical
           size. `None` means the whole tensor is independent of the source.

Semantics of the dependency propagation:

  elementwise          disjunction of operands
  reductions           disjunction along the reduced axes
  `Softmax(axis)`      disjunction along `axis`, an output element of a softmax
                       depends on every input element along its axis
  `MatMul(A, B)`       every entry of the contraction is assumed structurally
                       nonzero, so out[.., i, k] = any_j A[.., i, j] or
                       any_j B[.., j, k]
  `Conv`               every weight entry assumed nonzero, so the output
                       depends on the whole receptive field over all input
                       channels
  index operators      applied exactly
  `Where`              disjunction of the two data branches; the condition is
                       graph derived and contributes nothing

Assuming every weight entry nonzero makes the computed set an
OVER-APPROXIMATION of the true dependency set. The radius it yields is
therefore an upper bound on architectural support: the largest distance the
operator sequence can carry any influence at all.

Optional mask-aware mode additionally tracks which pre-softmax logits are
pinned to a large negative constant by the shifted-window attention mask and
removes those paths, giving the tighter architectural bound that the shifted
window design intends.
"""
from __future__ import annotations

import collections
import time

import numpy as np

import onnx_skim as osk

CONST_CAP = 20_000_000          # elements; larger constants are kept opaque
MASK_CONST_CAP = 200_000_000    # mask-aware mode needs the attention mask value
# The shifted-window mask in this export fills blocked logit pairs with -100.0
# and leaves permitted pairs at 0.0, so any cut between the two separates them.
ANN_THRESH = -50.0


class Val:
    __slots__ = ("shape", "dtype", "const", "dep", "ann")

    def __init__(self, shape, dtype, const=None, dep=None, ann=None):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.const = const
        self.dep = dep
        self.ann = ann

    @property
    def size(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def __repr__(self):
        return (f"Val{self.shape} const={'y' if self.const is not None else 'n'} "
                f"dep={None if self.dep is None else self.dep.shape}")


# ---------------------------------------------------------------------------
# broadcast-compressed boolean helpers
# ---------------------------------------------------------------------------
def _align(dep, ndim):
    """Left-pad a stored dep array with singleton axes to `ndim` dims."""
    if dep is None:
        return None
    if dep.ndim == ndim:
        return dep
    return dep.reshape((1,) * (ndim - dep.ndim) + dep.shape)


def bor(deps, out_shape):
    """Disjunction of several broadcast-compressed dep arrays."""
    live = [d for d in deps if d is not None and d.any()]
    if not live:
        return None
    nd = len(out_shape)
    acc = _align(live[0], nd)
    for d in live[1:]:
        acc = acc | _align(d, nd)
    # never let a stored axis exceed its logical size
    if acc.ndim != nd:
        acc = _align(acc, nd)
    return acc


def dep_full(dep, shape):
    """Materialise a broadcast-compressed dep to its logical shape."""
    if dep is None:
        return np.zeros(shape, dtype=bool)
    return np.broadcast_to(_align(dep, len(shape)), shape)


def split_groups(s_in, s_out):
    """Contiguous axis groups with matching element products."""
    n, m = len(s_in), len(s_out)
    i = j = 0
    groups = []
    while i < n or j < m:
        ia, ja = [], []
        while i < n and s_in[i] == 1:
            ia.append(i)
            i += 1
        while j < m and s_out[j] == 1:
            ja.append(j)
            j += 1
        if i >= n and j >= m:
            if ia or ja:
                groups.append((ia, ja))
            break
        pi = pj = 1
        if i < n:
            ia.append(i)
            pi = s_in[i]
            i += 1
        if j < m:
            ja.append(j)
            pj = s_out[j]
            j += 1
        while pi != pj:
            if pi < pj:
                pi *= s_in[i]
                ia.append(i)
                i += 1
            else:
                pj *= s_out[j]
                ja.append(j)
                j += 1
        groups.append((ia, ja))
    return groups


def dep_reshape(dep, s_in, s_out):
    """Reshape a broadcast-compressed dep, preserving compression per group."""
    if dep is None:
        return None
    d = _align(dep, len(s_in))
    tmp_in = list(d.shape)
    tmp_out = []
    for ia, ja in split_groups(s_in, s_out):
        allb = all(d.shape[a] == 1 for a in ia)
        if allb:
            for a in ia:
                tmp_in[a] = 1
            tmp_out += [1] * len(ja)
        else:
            for a in ia:
                tmp_in[a] = s_in[a]
            tmp_out += [s_out[a] for a in ja]
    x = np.broadcast_to(d, tuple(tmp_in))
    return np.ascontiguousarray(x).reshape(tuple(tmp_out))


def dep_expand_axis(dep, s, axis):
    """Materialise one broadcast axis of a dep array to its logical size."""
    if dep is None:
        return None
    d = _align(dep, len(s))
    if d.shape[axis] == s[axis]:
        return d
    tgt = list(d.shape)
    tgt[axis] = s[axis]
    return np.ascontiguousarray(np.broadcast_to(d, tuple(tgt)))


def dep_any(dep, axes, keepdims, s):
    if dep is None:
        return None
    d = _align(dep, len(s))
    return d.any(axis=tuple(axes), keepdims=bool(keepdims))


# ---------------------------------------------------------------------------
# shape helpers
# ---------------------------------------------------------------------------
def bshape(*shapes):
    nd = max(len(s) for s in shapes)
    out = []
    for k in range(nd):
        v = 1
        for s in shapes:
            p = (1,) * (nd - len(s)) + tuple(s)
            v = max(v, p[k])
        out.append(v)
    return tuple(out)


_ELEMENTWISE2 = {"Add", "Sub", "Mul", "Div", "Pow", "Equal", "And", "Or",
                 "Less", "Greater", "Min", "Max", "Mod"}
_ELEMENTWISE1 = {"Sqrt", "Erf", "Not", "Neg", "Abs", "Exp", "Log", "Relu",
                 "Sigmoid", "Tanh", "Identity", "Reciprocal", "Floor", "Ceil"}

def _idiv(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        q = np.abs(a) // np.abs(b)
        return (np.sign(a) * np.sign(b) * q).astype(np.result_type(a, b))
    return np.divide(a, b)


_NPOP2 = {
    "Add": np.add, "Sub": np.subtract, "Mul": np.multiply,
    "Div": _idiv,
    "Pow": np.power, "Equal": np.equal, "And": np.logical_and,
    "Or": np.logical_or, "Less": np.less, "Greater": np.greater,
    "Min": np.minimum, "Max": np.maximum, "Mod": np.mod,
}
_NPOP1 = {
    "Sqrt": np.sqrt, "Erf": None, "Not": np.logical_not, "Neg": np.negative,
    "Abs": np.abs, "Exp": np.exp, "Log": np.log, "Relu": None,
    "Sigmoid": None, "Tanh": np.tanh, "Identity": lambda x: x,
    "Reciprocal": np.reciprocal, "Floor": np.floor, "Ceil": np.ceil,
}

_ONNX_NP = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
            6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16,
            11: np.float64, 12: np.uint32, 13: np.uint64}


class Interp:
    def __init__(self, graph, source_ij, mask_aware=False, verbose=False):
        self.g = graph
        self.src = source_ij
        self.mask_aware = bool(mask_aware)
        self.cap = MASK_CONST_CAP if mask_aware else CONST_CAP
        self.verbose = verbose
        self.env = {}
        self.stats = collections.Counter()

    # -- entry --------------------------------------------------------------
    def run(self):
        g = self.g
        for name, t in g.initializer.items():
            const = t.load() if t.size <= self.cap else None
            self.env[name] = Val(t.dims, _ONNX_NP.get(t.data_type, np.float32),
                                 const=const)
        i, j = self.src
        for vi in g.inputs:
            sh = tuple(vi["shape"])
            d = np.zeros((1,) * (len(sh) - 2) + sh[-2:], dtype=bool)
            d[..., i, j] = True
            self.env[vi["name"]] = Val(sh, np.float32, dep=d)

        consumers = collections.Counter()
        for n in g.nodes:
            for x in n.input:
                if x:
                    consumers[x] += 1
        keep = {v["name"] for v in g.outputs}

        t0 = time.time()
        for k, n in enumerate(g.nodes):
            try:
                self.eval_node(n)
            except Exception as exc:                       # pragma: no cover
                raise RuntimeError(
                    f"node {k} {n.op_type} {n.input} -> {n.output}: {exc}") from exc
            self.stats[n.op_type] += 1
            for x in n.input:
                if not x or x in keep:
                    continue
                consumers[x] -= 1
                if consumers[x] <= 0:
                    self.env.pop(x, None)
            if self.verbose and k % 1000 == 0:
                print(f"  node {k}/{len(g.nodes)} {n.op_type} "
                      f"{time.time() - t0:.1f}s", flush=True)
        return {v["name"]: self.env[v["name"]] for v in g.outputs}

    # -- helpers ------------------------------------------------------------
    def _v(self, name):
        if name not in self.env:
            raise KeyError(f"missing tensor {name!r}")
        return self.env[name]

    def _need(self, name):
        v = self._v(name)
        if v.const is None:
            raise ValueError(f"tensor {name!r} needed as a concrete value but is opaque")
        return np.asarray(v.const)

    @staticmethod
    def _ai(n, key, default=None):
        a = n.attr.get(key)
        if a is None:
            return default
        if a.ints is not None:
            return list(a.ints)
        if a.i is not None:
            return a.i
        if a.f is not None:
            return a.f
        if a.s is not None:
            return a.s
        return default

    def _put(self, name, val):
        self.env[name] = val

    # -- dispatcher ---------------------------------------------------------
    def eval_node(self, n):
        op = n.op_type
        fn = getattr(self, f"op_{op}", None)
        if fn is not None:
            return fn(n)
        if op in _ELEMENTWISE2:
            return self.ew2(n)
        if op in _ELEMENTWISE1:
            return self.ew1(n)
        raise NotImplementedError(op)

    # -- operators ----------------------------------------------------------
    def op_Constant(self, n):
        a = n.attr.get("value")
        if a is None or a.t is None:
            raise NotImplementedError("Constant without a tensor value")
        t = a.t
        const = t.load() if t.size <= self.cap else None
        self._put(n.output[0], Val(t.dims, _ONNX_NP.get(t.data_type, np.float32),
                                   const=const))

    def op_Shape(self, n):
        v = self._v(n.input[0])
        self._put(n.output[0], Val((len(v.shape),), np.int64,
                                   const=np.array(v.shape, dtype=np.int64)))

    def op_ConstantOfShape(self, n):
        sh = tuple(int(x) for x in self._need(n.input[0]))
        a = n.attr.get("value")
        if a is not None and a.t is not None:
            fill = a.t.load().ravel()[0]
            dt = _ONNX_NP.get(a.t.data_type, np.float32)
        else:
            fill, dt = np.float32(0.0), np.float32
        size = int(np.prod(sh)) if sh else 1
        const = np.full(sh, fill, dtype=dt) if size <= self.cap else None
        self._put(n.output[0], Val(sh, dt, const=const))

    def op_Cast(self, n):
        v = self._v(n.input[0])
        dt = _ONNX_NP.get(int(self._ai(n, "to", 1)), np.float32)
        const = v.const.astype(dt) if v.const is not None else None
        self._put(n.output[0], Val(v.shape, dt, const=const, dep=v.dep, ann=v.ann))

    def ew1(self, n):
        v = self._v(n.input[0])
        op = n.op_type
        const = None
        if v.const is not None:
            f = _NPOP1[op]
            if f is not None:
                with np.errstate(all="ignore"):
                    const = f(v.const)
            elif op == "Erf":
                const = None                    # value never needed downstream
        dt = np.bool_ if op == "Not" else v.dtype
        self._put(n.output[0], Val(v.shape, dt, const=const, dep=v.dep, ann=v.ann))

    def ew2(self, n):
        a, b = self._v(n.input[0]), self._v(n.input[1])
        out = bshape(a.shape, b.shape)
        op = n.op_type
        dt = np.bool_ if op in ("Equal", "Less", "Greater", "And", "Or") else \
            (a.dtype if a.const is None or b.const is None else
             np.result_type(a.dtype, b.dtype))
        const = None
        if a.const is not None and b.const is not None and \
                int(np.prod(out)) <= self.cap:
            with np.errstate(all="ignore"):
                const = _NPOP2[op](a.const, b.const)
            self._put(n.output[0], Val(out, const.dtype, const=const))
            return
        dep = bor([a.dep, b.dep], out)
        ann = bor([a.ann, b.ann], out)
        if self.mask_aware and op == "Add":
            for other in (b, a):
                if other.const is not None and other.const.dtype.kind == "f":
                    m = np.asarray(other.const) <= ANN_THRESH
                    if m.any():
                        ann = bor([ann, m], out)
                        self.stats["_attention_mask_applied"] += 1
                        self.stats["_attention_mask_blocked_entries"] += int(m.sum())
        self._put(n.output[0], Val(out, dt, dep=dep, ann=ann))

    def op_Where(self, n):
        c, x, y = (self._v(t) for t in n.input)
        out = bshape(c.shape, x.shape, y.shape)
        if c.const is not None and x.const is not None and y.const is not None \
                and int(np.prod(out)) <= self.cap:
            const = np.where(c.const, x.const, y.const)
            self._put(n.output[0], Val(out, const.dtype, const=const))
            return
        self._put(n.output[0], Val(out, x.dtype,
                                   dep=bor([x.dep, y.dep], out),
                                   ann=bor([x.ann, y.ann], out)))

    def op_Reshape(self, n):
        v = self._v(n.input[0])
        tgt = [int(x) for x in self._need(n.input[1])]
        allowzero = int(self._ai(n, "allowzero", 0))
        out = list(tgt)
        for k, s in enumerate(out):
            if s == 0 and not allowzero:
                out[k] = v.shape[k]
        if -1 in out:
            k = out.index(-1)
            rest = 1
            for q, s in enumerate(out):
                if q != k:
                    rest *= s
            out[k] = v.size // max(rest, 1)
        out = tuple(out)
        if v.const is not None:
            self._put(n.output[0], Val(out, v.dtype, const=v.const.reshape(out)))
            return
        self._put(n.output[0], Val(out, v.dtype,
                                   dep=dep_reshape(v.dep, v.shape, out),
                                   ann=dep_reshape(v.ann, v.shape, out)))

    def op_Transpose(self, n):
        v = self._v(n.input[0])
        perm = self._ai(n, "perm", list(range(len(v.shape))[::-1]))
        out = tuple(v.shape[p] for p in perm)
        if v.const is not None:
            self._put(n.output[0], Val(out, v.dtype, const=v.const.transpose(perm)))
            return
        f = (lambda d: None if d is None
             else _align(d, len(v.shape)).transpose(perm))
        self._put(n.output[0], Val(out, v.dtype, dep=f(v.dep), ann=f(v.ann)))

    def op_Unsqueeze(self, n):
        v = self._v(n.input[0])
        axes = [int(x) for x in self._need(n.input[1])]
        nd = len(v.shape) + len(axes)
        axes = sorted(a % nd for a in axes)
        out = list(v.shape)
        for a in axes:
            out.insert(a, 1)
        out = tuple(out)
        if v.const is not None:
            self._put(n.output[0], Val(out, v.dtype, const=v.const.reshape(out)))
            return
        f = (lambda d: None if d is None
             else dep_reshape(d, v.shape, out))
        self._put(n.output[0], Val(out, v.dtype, dep=f(v.dep), ann=f(v.ann)))

    def op_Squeeze(self, n):
        v = self._v(n.input[0])
        if len(n.input) > 1 and n.input[1]:
            axes = sorted(int(x) % len(v.shape) for x in self._need(n.input[1]))
        else:
            axes = [k for k, s in enumerate(v.shape) if s == 1]
        out = tuple(s for k, s in enumerate(v.shape) if k not in axes)
        if v.const is not None:
            self._put(n.output[0], Val(out, v.dtype, const=v.const.reshape(out)))
            return
        f = lambda d: None if d is None else dep_reshape(d, v.shape, out)
        self._put(n.output[0], Val(out, v.dtype, dep=f(v.dep), ann=f(v.ann)))

    def op_Concat(self, n):
        vs = [self._v(t) for t in n.input]
        ax = int(self._ai(n, "axis", 0)) % len(vs[0].shape)
        out = list(vs[0].shape)
        out[ax] = sum(v.shape[ax] for v in vs)
        out = tuple(out)
        if all(v.const is not None for v in vs) and int(np.prod(out)) <= self.cap:
            self._put(n.output[0], Val(out, vs[0].dtype,
                                       const=np.concatenate([v.const for v in vs],
                                                            axis=ax)))
            return

        def cat(field):
            arrs = [getattr(v, field) for v in vs]
            if all(a is None for a in arrs):
                return None
            nd = len(out)
            tgt = [1] * nd
            for v, a in zip(vs, arrs):
                if a is None:
                    continue
                a = _align(a, nd)
                for k in range(nd):
                    if k != ax:
                        tgt[k] = max(tgt[k], a.shape[k])
            parts = []
            for v, a in zip(vs, arrs):
                sh = list(tgt)
                sh[ax] = v.shape[ax]
                if a is None:
                    parts.append(np.zeros(sh, dtype=bool))
                else:
                    parts.append(np.broadcast_to(_align(a, nd), sh))
            return np.concatenate(parts, axis=ax)

        self._put(n.output[0], Val(out, vs[0].dtype, dep=cat("dep"), ann=cat("ann")))

    def op_Slice(self, n):
        v = self._v(n.input[0])
        starts = [int(x) for x in self._need(n.input[1])]
        ends = [int(x) for x in self._need(n.input[2])]
        axes = ([int(x) for x in self._need(n.input[3])]
                if len(n.input) > 3 and n.input[3] else list(range(len(starts))))
        steps = ([int(x) for x in self._need(n.input[4])]
                 if len(n.input) > 4 and n.input[4] else [1] * len(starts))
        nd = len(v.shape)
        sl = [slice(None)] * nd
        out = list(v.shape)
        for st, en, ax, sp in zip(starts, ends, axes, steps):
            ax %= nd
            dim = v.shape[ax]
            if sp > 0:
                a = min(max(st if st >= 0 else st + dim, 0), dim)
                b = min(max(en if en >= 0 else en + dim, 0), dim)
                sl[ax] = slice(a, b, sp)
                out[ax] = max(0, (b - a + sp - 1) // sp)
            else:
                a = min(max(st if st >= 0 else st + dim, 0), dim - 1)
                b = min(max(en if en >= 0 else en + dim, -1), dim - 1)
                sl[ax] = slice(a, None if b < 0 else b, sp)
                out[ax] = max(0, (a - b + (-sp) - 1) // (-sp))
        out = tuple(out)
        if v.const is not None:
            c = v.const[tuple(sl)]
            self._put(n.output[0], Val(c.shape, v.dtype, const=c))
            return

        def cut(d):
            if d is None:
                return None
            d = _align(d, nd)
            s2 = list(sl)
            for k in range(nd):
                if d.shape[k] == 1 and v.shape[k] != 1:
                    s2[k] = slice(None)
            r = d[tuple(s2)]
            return r

        self._put(n.output[0], Val(out, v.dtype, dep=cut(v.dep), ann=cut(v.ann)))

    def op_Gather(self, n):
        v = self._v(n.input[0])
        idx = self._need(n.input[1])
        ax = int(self._ai(n, "axis", 0)) % len(v.shape)
        idx = np.asarray(idx)
        out = v.shape[:ax] + tuple(idx.shape) + v.shape[ax + 1:]
        if v.const is not None:
            c = np.take(v.const, idx, axis=ax)
            self._put(n.output[0], Val(c.shape, v.dtype, const=c))
            return

        def take(d):
            if d is None:
                return None
            d = _align(d, len(v.shape))
            if d.shape[ax] == 1:
                return d.reshape(d.shape[:ax] + (1,) * idx.ndim + d.shape[ax + 1:])
            return np.take(d, idx, axis=ax)

        self._put(n.output[0], Val(out, v.dtype, dep=take(v.dep), ann=take(v.ann)))

    def op_Expand(self, n):
        v = self._v(n.input[0])
        sh = tuple(int(x) for x in self._need(n.input[1]))
        out = bshape(v.shape, sh)
        if v.const is not None and int(np.prod(out)) <= self.cap:
            self._put(n.output[0], Val(out, v.dtype,
                                       const=np.broadcast_to(v.const, out).copy()))
            return
        f = lambda d: None if d is None else _align(d, len(out))
        self._put(n.output[0], Val(out, v.dtype, dep=f(v.dep), ann=f(v.ann)))

    def op_Pad(self, n):
        v = self._v(n.input[0])
        pads = [int(x) for x in self._need(n.input[1])]
        nd = len(v.shape)
        beg, end = pads[:nd], pads[nd:]
        out = tuple(v.shape[k] + beg[k] + end[k] for k in range(nd))
        if v.const is not None and int(np.prod(out)) <= self.cap:
            cv = 0
            if len(n.input) > 2 and n.input[2]:
                cv = self._need(n.input[2]).ravel()[0]
            c = np.pad(v.const, list(zip(beg, end)), constant_values=cv)
            self._put(n.output[0], Val(out, v.dtype, const=c))
            return

        def pd(d):
            if d is None:
                return None
            d = _align(d, nd)
            for k in range(nd):
                if (beg[k] or end[k]) and d.shape[k] != v.shape[k]:
                    d = dep_expand_axis(d, v.shape, k)
            widths = [(beg[k], end[k]) if d.shape[k] == v.shape[k] else (0, 0)
                      for k in range(nd)]
            return np.pad(d, widths, constant_values=False)

        self._put(n.output[0], Val(out, v.dtype, dep=pd(v.dep), ann=pd(v.ann)))

    def op_ScatterND(self, n):
        data, idx, upd = (self._v(t) for t in n.input)
        out = data.shape
        if data.const is not None and idx.const is not None and \
                upd.const is not None and int(np.prod(out)) <= self.cap:
            c = np.array(data.const, copy=True)
            ind = np.asarray(idx.const).reshape(-1, idx.shape[-1])
            u = np.asarray(upd.const).reshape(ind.shape[0], -1)
            c[tuple(ind.T)] = u.reshape(c[tuple(ind.T)].shape)
            self._put(n.output[0], Val(out, data.dtype, const=c))
            return
        self._put(n.output[0], Val(out, data.dtype,
                                   dep=bor([data.dep, upd.dep], out),
                                   ann=bor([data.ann, upd.ann], out)))

    def op_ReduceMean(self, n):
        v = self._v(n.input[0])
        axes = self._ai(n, "axes")
        if axes is None and len(n.input) > 1 and n.input[1]:
            axes = [int(x) for x in self._need(n.input[1])]
        if axes is None:
            axes = list(range(len(v.shape)))
        keep = int(self._ai(n, "keepdims", 1))
        axes = [a % len(v.shape) for a in axes]
        out = tuple(s if k not in axes else 1
                    for k, s in enumerate(v.shape)) if keep else \
            tuple(s for k, s in enumerate(v.shape) if k not in axes)
        if v.const is not None:
            c = v.const.mean(axis=tuple(axes), keepdims=bool(keep))
            self._put(n.output[0], Val(out, v.dtype, const=c))
            return
        self._put(n.output[0], Val(out, v.dtype,
                                   dep=dep_any(v.dep, axes, keep, v.shape),
                                   ann=None))

    def op_Softmax(self, n):
        v = self._v(n.input[0])
        ax = int(self._ai(n, "axis", -1)) % len(v.shape)
        if v.dep is None:
            self._put(n.output[0], Val(v.shape, v.dtype, dep=None, ann=v.ann))
            return
        d = _align(v.dep, len(v.shape))
        if self.mask_aware and v.ann is not None:
            a = _align(v.ann, len(v.shape))
            d = dep_expand_axis(d, v.shape, ax) & ~np.broadcast_to(
                a, dep_expand_axis(d, v.shape, ax).shape)
        red = d.any(axis=ax, keepdims=True)
        self._put(n.output[0], Val(v.shape, v.dtype, dep=red, ann=v.ann))

    def op_MatMul(self, n):
        a, b = self._v(n.input[0]), self._v(n.input[1])
        sa, sb = a.shape, b.shape
        m, k1 = sa[-2], sa[-1]
        k2, p = sb[-2], sb[-1]
        if k1 != k2:
            raise ValueError(f"MatMul contraction {sa} x {sb}")
        batch = bshape(sa[:-2], sb[:-2])
        out = batch + (m, p)
        da = _align(a.dep, len(sa)) if a.dep is not None else None
        db = _align(b.dep, len(sb)) if b.dep is not None else None
        ann = _align(a.ann, len(sa)) if (self.mask_aware and a.ann is not None) else None

        terms = []
        if da is not None:
            if ann is None:
                terms.append(da.any(axis=-1, keepdims=True))          # (.., m, 1)
            else:
                full = dep_expand_axis(da, sa, len(sa) - 1)
                keepj = ~np.broadcast_to(_align(ann, full.ndim), full.shape)
                terms.append((full & keepj).any(axis=-1, keepdims=True))
        if db is not None:
            if ann is None:
                terms.append(db.any(axis=-2, keepdims=True))           # (.., 1, p)
            elif db.shape[-1] == 1:
                # out[.., i, q] = any_j ( not ann[.., i, j] and db[.., j, 0] )
                row = np.swapaxes(db, -1, -2)                          # (.., 1, k)
                keepj = ~_align(ann, max(ann.ndim, row.ndim))
                terms.append((keepj & _align(row, keepj.ndim)
                              ).any(axis=-1, keepdims=True))
            else:
                self.stats["_matmul_mask_fallback"] += 1
                terms.append(db.any(axis=-2, keepdims=True))
        dep = bor(terms, out) if terms else None
        self._put(n.output[0], Val(out, a.dtype, dep=dep))

    def op_Conv(self, n):
        x, w = self._v(n.input[0]), self._v(n.input[1])
        nd = len(x.shape) - 2
        ks = self._ai(n, "kernel_shape", list(w.shape[2:]))
        st = self._ai(n, "strides", [1] * nd)
        dl = self._ai(n, "dilations", [1] * nd)
        pd = self._ai(n, "pads", [0] * (2 * nd))
        cout = w.shape[0]
        osp = []
        for k in range(nd):
            eff = dl[k] * (ks[k] - 1) + 1
            osp.append((x.shape[2 + k] + pd[k] + pd[nd + k] - eff) // st[k] + 1)
        out = (x.shape[0], cout) + tuple(osp)
        if x.dep is None:
            self._put(n.output[0], Val(out, x.dtype, dep=None))
            return
        d = _align(x.dep, len(x.shape))
        d = d.any(axis=1, keepdims=True)                    # all input channels mix
        for k in range(nd):
            if d.shape[2 + k] == 1:
                continue
            d = np.pad(d, [(0, 0)] * (2 + k) + [(pd[k], pd[nd + k])] +
                       [(0, 0)] * (nd - k - 1), constant_values=False)
        acc = None
        offs = np.ndindex(*ks)
        for off in offs:
            sl = [slice(None), slice(None)]
            for k in range(nd):
                if d.shape[2 + k] == 1:
                    sl.append(slice(None))
                else:
                    a0 = off[k] * dl[k]
                    sl.append(slice(a0, a0 + st[k] * (osp[k] - 1) + 1, st[k]))
            piece = d[tuple(sl)]
            acc = piece if acc is None else (acc | piece)
        self._put(n.output[0], Val(out, x.dtype, dep=acc))
