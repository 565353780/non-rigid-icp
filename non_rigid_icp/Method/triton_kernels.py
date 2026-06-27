"""Optional Triton acceleration for the dense, regular numerical kernels.

These back the per-element-parallel hot paths of the trajectory guard and the
clamped step. They are *optional*: if Triton (or CUDA) is unavailable, or the
tensors live on CPU, every entry point transparently falls back to the existing
vectorized torch implementation, and the results are required to match the torch
path (the unit tests assert this). Only regular, batched arithmetic kernels are
Triton-ised here -- topology rebuilds, I/O and the Open3D/Embree closest-point
queries are left to their existing backends by design.

Kernels:
  segmentTrianglePairHits : per candidate-(segment, triangle) Moller-Trumbore
                            pierce test, fused so the (segment, tri) gather +
                            the cross/dot arithmetic happen in one pass (the torch
                            version materializes several (P, 3) intermediates).
  segmentTrianglePairParams: same fused Moller-Trumbore, but emits the crossing
                            parameter ``t`` (inf for non-hits) so the caller can
                            reduce to each segment's EARLIEST crossing and pull it
                            back by the minimum distance, no bisection needed.
  clampNorm               : per-row vector magnitude clamp (the per-vertex step
                            cap of the clamped GD update).
"""

import torch
from typing import Union

try:  # pragma: no cover - availability depends on the runtime
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False


# Triton may import fine yet fail to JIT/launch at runtime (e.g. the CUDA driver
# stub `libcuda.so` is missing from the linker path). We probe lazily on the
# first launch and, on ANY failure, permanently fall back to the torch path for
# the rest of the process -- correctness is identical, only speed differs.
_TRITON_RUNTIME_OK = _HAS_TRITON


def tritonAvailable() -> bool:
    return _HAS_TRITON and _TRITON_RUNTIME_OK


def _disableTritonRuntime(err: Exception) -> None:
    global _TRITON_RUNTIME_OK
    if _TRITON_RUNTIME_OK:
        print(
            "[WARN][triton_kernels] Triton launch failed "
            f"({type(err).__name__}: {err}); falling back to the torch path "
            "for the rest of this run."
        )
    _TRITON_RUNTIME_OK = False


# --------------------------------------------------------------------------- #
# segment-triangle pierce test over candidate pairs                           #
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _seg_tri_pair_kernel(
        seg_id_ptr, tri_id_ptr,
        ss_ptr, se_ptr,            # (S, 3) segment start/end
        va_ptr, vb_ptr, vc_ptr,    # (T, 3) triangle a/b/c
        out_ptr,
        P, eps,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < P

        sid = tl.load(seg_id_ptr + offs, mask=mask, other=0)
        tid = tl.load(tri_id_ptr + offs, mask=mask, other=0)

        # gather segment endpoints (row-major, 3 cols)
        px = tl.load(ss_ptr + sid * 3 + 0, mask=mask, other=0.0)
        py = tl.load(ss_ptr + sid * 3 + 1, mask=mask, other=0.0)
        pz = tl.load(ss_ptr + sid * 3 + 2, mask=mask, other=0.0)
        qx = tl.load(se_ptr + sid * 3 + 0, mask=mask, other=0.0)
        qy = tl.load(se_ptr + sid * 3 + 1, mask=mask, other=0.0)
        qz = tl.load(se_ptr + sid * 3 + 2, mask=mask, other=0.0)

        ax = tl.load(va_ptr + tid * 3 + 0, mask=mask, other=0.0)
        ay = tl.load(va_ptr + tid * 3 + 1, mask=mask, other=0.0)
        az = tl.load(va_ptr + tid * 3 + 2, mask=mask, other=0.0)
        bx = tl.load(vb_ptr + tid * 3 + 0, mask=mask, other=0.0)
        by = tl.load(vb_ptr + tid * 3 + 1, mask=mask, other=0.0)
        bz = tl.load(vb_ptr + tid * 3 + 2, mask=mask, other=0.0)
        cx = tl.load(vc_ptr + tid * 3 + 0, mask=mask, other=0.0)
        cy = tl.load(vc_ptr + tid * 3 + 1, mask=mask, other=0.0)
        cz = tl.load(vc_ptr + tid * 3 + 2, mask=mask, other=0.0)

        dx = qx - px
        dy = qy - py
        dz = qz - pz
        e1x = bx - ax
        e1y = by - ay
        e1z = bz - az
        e2x = cx - ax
        e2y = cy - ay
        e2z = cz - az

        # pvec = d x e2
        pvx = dy * e2z - dz * e2y
        pvy = dz * e2x - dx * e2z
        pvz = dx * e2y - dy * e2x
        det = e1x * pvx + e1y * pvy + e1z * pvz

        parallel = tl.abs(det) < eps
        inv_det = 1.0 / tl.where(parallel, tl.full(det.shape, 1.0, det.dtype), det)

        tvx = px - ax
        tvy = py - ay
        tvz = pz - az
        u = (tvx * pvx + tvy * pvy + tvz * pvz) * inv_det

        # qvec = tvec x e1
        qvx = tvy * e1z - tvz * e1y
        qvy = tvz * e1x - tvx * e1z
        qvz = tvx * e1y - tvy * e1x
        v = (dx * qvx + dy * qvy + dz * qvz) * inv_det
        t = (e2x * qvx + e2y * qvy + e2z * qvz) * inv_det

        hit = (
            (~parallel)
            & (u >= -eps)
            & (v >= -eps)
            & (u + v <= 1.0 + eps)
            & (t >= -eps)
            & (t <= 1.0 + eps)
        )
        tl.store(out_ptr + offs, hit.to(tl.int8), mask=mask)


def segmentTrianglePairHits(
    seg_id: torch.Tensor,
    tri_id: torch.Tensor,
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    tri_a: torch.Tensor,
    tri_b: torch.Tensor,
    tri_c: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """(P,) bool: does candidate pair (seg_id[i], tri_id[i]) pierce?

    `seg_start/seg_end` are (S, 3) per-segment endpoints; `tri_a/b/c` are (T, 3)
    per-triangle vertices; `seg_id/tri_id` are (P,) indices into them. Triton
    path on CUDA, exact torch Moller-Trumbore fallback otherwise."""
    p = seg_id.shape[0]
    if p == 0:
        return torch.zeros(0, dtype=torch.bool, device=seg_id.device)

    def _torch():
        from non_rigid_icp.Method.geometry import segmentTriangleIntersect
        return segmentTriangleIntersect(
            seg_start[seg_id], seg_end[seg_id],
            tri_a[tri_id], tri_b[tri_id], tri_c[tri_id],
            eps=eps,
        )

    if not (tritonAvailable() and seg_id.is_cuda):
        return _torch()
    try:
        out = torch.empty(p, dtype=torch.int8, device=seg_id.device)
        BLOCK = 1024
        grid = (triton.cdiv(p, BLOCK),)
        _seg_tri_pair_kernel[grid](
            seg_id.contiguous(), tri_id.contiguous(),
            seg_start.contiguous(), seg_end.contiguous(),
            tri_a.contiguous(), tri_b.contiguous(), tri_c.contiguous(),
            out, p, eps, BLOCK=BLOCK,
        )
        return out.bool()
    except Exception as e:  # pragma: no cover - environment dependent
        _disableTritonRuntime(e)
        return _torch()


# --------------------------------------------------------------------------- #
# segment-triangle pierce parameter (t) over candidate pairs                  #
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _seg_tri_param_kernel(
        seg_id_ptr, tri_id_ptr,
        ss_ptr, se_ptr,            # (S, 3) segment start/end
        va_ptr, vb_ptr, vc_ptr,    # (T, 3) triangle a/b/c
        t_out_ptr,                 # (P,) earliest-crossing param (inf if no hit)
        P, eps, t_lo, t_hi,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < P

        sid = tl.load(seg_id_ptr + offs, mask=mask, other=0)
        tid = tl.load(tri_id_ptr + offs, mask=mask, other=0)

        px = tl.load(ss_ptr + sid * 3 + 0, mask=mask, other=0.0)
        py = tl.load(ss_ptr + sid * 3 + 1, mask=mask, other=0.0)
        pz = tl.load(ss_ptr + sid * 3 + 2, mask=mask, other=0.0)
        qx = tl.load(se_ptr + sid * 3 + 0, mask=mask, other=0.0)
        qy = tl.load(se_ptr + sid * 3 + 1, mask=mask, other=0.0)
        qz = tl.load(se_ptr + sid * 3 + 2, mask=mask, other=0.0)

        ax = tl.load(va_ptr + tid * 3 + 0, mask=mask, other=0.0)
        ay = tl.load(va_ptr + tid * 3 + 1, mask=mask, other=0.0)
        az = tl.load(va_ptr + tid * 3 + 2, mask=mask, other=0.0)
        bx = tl.load(vb_ptr + tid * 3 + 0, mask=mask, other=0.0)
        by = tl.load(vb_ptr + tid * 3 + 1, mask=mask, other=0.0)
        bz = tl.load(vb_ptr + tid * 3 + 2, mask=mask, other=0.0)
        cx = tl.load(vc_ptr + tid * 3 + 0, mask=mask, other=0.0)
        cy = tl.load(vc_ptr + tid * 3 + 1, mask=mask, other=0.0)
        cz = tl.load(vc_ptr + tid * 3 + 2, mask=mask, other=0.0)

        dx = qx - px
        dy = qy - py
        dz = qz - pz
        e1x = bx - ax
        e1y = by - ay
        e1z = bz - az
        e2x = cx - ax
        e2y = cy - ay
        e2z = cz - az

        pvx = dy * e2z - dz * e2y
        pvy = dz * e2x - dx * e2z
        pvz = dx * e2y - dy * e2x
        det = e1x * pvx + e1y * pvy + e1z * pvz

        parallel = tl.abs(det) < eps
        inv_det = 1.0 / tl.where(parallel, tl.full(det.shape, 1.0, det.dtype), det)

        tvx = px - ax
        tvy = py - ay
        tvz = pz - az
        u = (tvx * pvx + tvy * pvy + tvz * pvz) * inv_det

        qvx = tvy * e1z - tvz * e1y
        qvy = tvz * e1x - tvx * e1z
        qvz = tvx * e1y - tvy * e1x
        v = (dx * qvx + dy * qvy + dz * qvz) * inv_det
        t = (e2x * qvx + e2y * qvy + e2z * qvz) * inv_det

        hit = (
            (~parallel)
            & (u >= -eps)
            & (v >= -eps)
            & (u + v <= 1.0 + eps)
            & (t >= t_lo - eps)
            & (t <= t_hi + eps)
        )
        t_store = tl.where(hit, t, tl.full(t.shape, float("inf"), t.dtype))
        tl.store(t_out_ptr + offs, t_store, mask=mask)


def segmentTrianglePairParams(
    seg_id: torch.Tensor,
    tri_id: torch.Tensor,
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
    tri_a: torch.Tensor,
    tri_b: torch.Tensor,
    tri_c: torch.Tensor,
    eps: float = 1e-9,
    t_lo: float = 0.0,
    t_hi: float = 1.0,
) -> torch.Tensor:
    """(P,) segment parameter ``t`` of the pierce point for candidate pairs.

    Same layout/semantics as `segmentTrianglePairHits`, but returns the crossing
    parameter ``t`` (``+inf`` where the pair does not pierce inside ``[t_lo,
    t_hi]``) instead of a boolean. ``t = 0`` is the segment's rest endpoint and
    ``t = 1`` its current endpoint, so a per-segment ``amin`` over its pairs is
    the earliest (closest-to-rest) crossing. Triton path on CUDA, exact torch
    Moller-Trumbore fallback otherwise (identical result)."""
    p = seg_id.shape[0]
    if p == 0:
        return torch.zeros(0, device=seg_id.device)

    def _torch():
        from non_rigid_icp.Method.geometry import segmentTriangleIntersectParams
        _, t, _, _ = segmentTriangleIntersectParams(
            seg_start[seg_id], seg_end[seg_id],
            tri_a[tri_id], tri_b[tri_id], tri_c[tri_id],
            eps=eps, t_lo=t_lo, t_hi=t_hi,
        )
        return t

    if not (tritonAvailable() and seg_id.is_cuda):
        return _torch()
    try:
        out = torch.empty(p, dtype=torch.float32, device=seg_id.device)
        BLOCK = 1024
        grid = (triton.cdiv(p, BLOCK),)
        _seg_tri_param_kernel[grid](
            seg_id.contiguous(), tri_id.contiguous(),
            seg_start.contiguous(), seg_end.contiguous(),
            tri_a.contiguous(), tri_b.contiguous(), tri_c.contiguous(),
            out, p, eps, float(t_lo), float(t_hi), BLOCK=BLOCK,
        )
        return out
    except Exception as e:  # pragma: no cover - environment dependent
        _disableTritonRuntime(e)
        return _torch()


# --------------------------------------------------------------------------- #
# per-row vector norm clamp                                                   #
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _clamp_norm_kernel(
        v_ptr, cap_ptr, out_ptr, N, eps,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(v_ptr + offs * 3 + 0, mask=mask, other=0.0)
        y = tl.load(v_ptr + offs * 3 + 1, mask=mask, other=0.0)
        z = tl.load(v_ptr + offs * 3 + 2, mask=mask, other=0.0)
        cap = tl.load(cap_ptr + offs, mask=mask, other=0.0)
        norm = tl.sqrt(x * x + y * y + z * z)
        scale = tl.minimum(cap / (norm + eps), 1.0)
        tl.store(out_ptr + offs * 3 + 0, x * scale, mask=mask)
        tl.store(out_ptr + offs * 3 + 1, y * scale, mask=mask)
        tl.store(out_ptr + offs * 3 + 2, z * scale, mask=mask)


def clampNorm(
    vectors: torch.Tensor, max_norm: Union[torch.Tensor, float], eps: float = 1e-20
) -> torch.Tensor:
    """Clamp each row of `vectors` (N, 3) to length <= max_norm (per-row or
    scalar). Triton path on CUDA, torch fallback otherwise (identical result)."""
    n = vectors.shape[0]

    def _torch():
        norm = vectors.norm(dim=-1, keepdim=True)
        if isinstance(max_norm, torch.Tensor):
            cap = max_norm.reshape(-1, 1)
        else:
            cap = torch.as_tensor(
                float(max_norm), device=vectors.device
            ).reshape(1, 1)
        scale = torch.clamp(cap / (norm + eps), max=1.0)
        return vectors * scale

    if not (tritonAvailable() and vectors.is_cuda) or n == 0:
        return _torch()
    try:
        if isinstance(max_norm, torch.Tensor):
            cap = max_norm.reshape(-1).to(vectors.dtype).contiguous()
        else:
            cap = torch.full(
                (n,), float(max_norm), device=vectors.device, dtype=vectors.dtype
            )
        out = torch.empty_like(vectors)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _clamp_norm_kernel[grid](
            vectors.contiguous(), cap, out, n, eps, BLOCK=BLOCK
        )
        return out
    except Exception as e:  # pragma: no cover - environment dependent
        _disableTritonRuntime(e)
        return _torch()
