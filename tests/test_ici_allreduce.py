"""The four-chip one-shot reduction kernels at several rows per call (TPU v4-8 only).

Checks, for 8 to 40 rows (the TP decoder passes ``batch`` rows; 32 rows is the 192 KiB case):
every chip holds bitwise identical results; ``one_shot_psum`` equals the float32 sum in mesh order;
``one_shot_psum_norm`` (with the next norm, the per-layer-input block and the layer scalar) agrees with
a plain JAX version of the same rounding points up to the last bit; and a row's result does not depend
on how many rows the call has (the first 8 rows are bitwise equal across row counts). The fused kernel
is checked with both exchange types: float32, and bf16 (each chip's partial sum rounded to bf16 before it
is sent, the four rounded arrays added in float32 in mesh order). The two-round merge
(``algorithm="scatter_gather"``) must give the same bits as the one-shot merge.

The reference is compiled with ``xla_allow_excess_precision=False``. With XLA's default (True), XLA
drops the float32 -> bf16 -> float32 rounding pairs inside a fusion and keeps float32 until the fusion
output, so only 37-73% of the elements agree with the kernel, which rounds after every op. Without
excess precision the two agree except where Mosaic and XLA differ in the last float32 bit of a
reduction, a dot or ``rsqrt`` and the value sits on a bf16 rounding boundary; the flipped bit then
passes through the following rounded ops. Over 10 seeds at 8-40 rows: at most 1.5e-5 of the elements
differ after attention and 9e-4 after the per-layer-input block, by at most 2 and 4 bf16 steps measured
at max(|value|, row RMS), with no trend in the row count. The test allows 2e-3 and 2**-5 (8 steps);
a wrong slice or weight moves nearly every element by the size of the value itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import PartitionSpec as P

pytestmark = pytest.mark.tpu

COLS = 1536
WIDTH = 256
LAYERS = 5
LAYER = 3
EPS = 1e-6
ROWS = (8, 16, 32, 40)


@pytest.fixture(scope="module")
def mesh():
    if jax.devices()[0].platform != "tpu" or len(jax.devices()) != 4:
        pytest.skip("needs the four chips of a TPU v4-8")
    from gemma4_posttrain_jax.tp_rollout import make_tp_mesh

    return make_tp_mesh(1, 4, jax.devices())


def _inputs(rows: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    bf16 = jnp.bfloat16
    return {
        # each chip's float32 partial sum, stacked along the first axis in mesh order
        "partial": rng.standard_normal((4, rows, COLS)).astype(np.float32),
        "residual": np.asarray(jnp.asarray(rng.standard_normal((rows, COLS)), bf16)),
        "post": np.asarray(jnp.asarray(1 + 0.1 * rng.standard_normal(COLS), bf16)),
        "next": np.asarray(jnp.asarray(1 + 0.1 * rng.standard_normal(COLS), bf16)),
        "ple_post": np.asarray(jnp.asarray(1 + 0.1 * rng.standard_normal(COLS), bf16)),
        "scale": np.asarray(jnp.asarray(np.full(COLS, 0.75), bf16)),
        "gate": np.asarray(jnp.asarray(0.03 * rng.standard_normal((COLS, WIDTH)), bf16)),
        "projection": np.asarray(jnp.asarray(0.06 * rng.standard_normal((WIDTH, COLS)), bf16)),
        "inputs": np.asarray(jnp.asarray(rng.standard_normal((rows, LAYERS * WIDTH)), bf16)),
    }


def _assert_close_to_reference(actual: np.ndarray, expected: np.ndarray) -> None:
    """Few elements differ, each by a few bf16 steps at the scale of the value or of its row."""

    actual32, expected32 = actual.astype(np.float32), expected.astype(np.float32)
    differ = actual32 != expected32
    assert np.mean(differ) <= 2e-3, f"{np.sum(differ)} of {differ.size} elements differ"
    rms = np.sqrt(np.mean(expected32 * expected32, axis=-1, keepdims=True))
    scaled = np.abs(actual32 - expected32) / np.maximum(np.abs(expected32), rms)
    assert scaled.max() <= 2**-5, f"largest difference is {scaled.max():.4f} of max(|value|, row RMS)"


def _take_rows(values: dict[str, np.ndarray], rows: int) -> dict[str, np.ndarray]:
    taken = dict(values)
    taken["partial"] = values["partial"][:, :rows]
    taken["residual"] = values["residual"][:rows]
    taken["inputs"] = values["inputs"][:rows]
    return taken


def _run_psum(mesh, partial: np.ndarray) -> np.ndarray:
    from gemma4_posttrain_jax.pallas.ici_allreduce import one_shot_psum
    from gemma4_posttrain_jax.tp_rollout import DP_AXIS, TP_AXIS

    stacked = partial.reshape(-1, COLS)
    run = jax.jit(
        jax.shard_map(
            lambda x: one_shot_psum(x, TP_AXIS, other_axes=(DP_AXIS,)),
            mesh=mesh,
            in_specs=P(TP_AXIS),
            out_specs=P(TP_AXIS),
            check_vma=False,
        )
    )
    return np.asarray(run(jnp.asarray(stacked))).reshape(4, *partial.shape[1:])


def _run_norm(
    mesh, values: dict[str, np.ndarray], *, ple: bool, exchange: str = "float32", algorithm: str = "one_shot"
) -> tuple[np.ndarray, np.ndarray]:
    from gemma4_posttrain_jax.pallas.ici_allreduce import PerLayerInput, one_shot_psum_norm
    from gemma4_posttrain_jax.tp_rollout import DP_AXIS, TP_AXIS

    def body(partial, residual, post, nxt, ple_post, scale, gate, projection, inputs):
        block = PerLayerInput(gate, projection, inputs, LAYER, ple_post) if ple else None
        return one_shot_psum_norm(
            partial,
            residual,
            post,
            nxt,
            eps=EPS,
            axis_name=TP_AXIS,
            other_axes=(DP_AXIS,),
            per_layer_input=block,
            scale=scale if ple else None,
            exchange_dtype=jnp.dtype(exchange),
            algorithm=algorithm,
        )

    names = ("residual", "post", "next", "ple_post", "scale", "gate", "projection", "inputs")
    run = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(TP_AXIS),) + (P(),) * len(names),
            out_specs=(P(TP_AXIS), P(TP_AXIS)),
            check_vma=False,
        )
    )
    rows = values["residual"].shape[0]
    x, h = run(jnp.asarray(values["partial"].reshape(-1, COLS)), *(jnp.asarray(values[name]) for name in names))
    return np.asarray(x).reshape(4, rows, COLS), np.asarray(h).reshape(4, rows, COLS)


def _reference(values: dict[str, np.ndarray], *, ple: bool, exchange: str = "float32") -> tuple[np.ndarray, np.ndarray]:
    """The kernel's arithmetic as plain JAX ops (float32 math, rounded to bf16 at the same points)."""

    dtype = jnp.bfloat16

    def rounded(value):
        return value.astype(dtype).astype(jnp.float32)

    def norm(value, weight):
        factor = lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + jnp.float32(EPS))
        return rounded(value * factor * jnp.asarray(weight).astype(jnp.float32))

    def compute(partial, residual):
        parts = [rounded(part) for part in partial] if exchange == "bfloat16" else list(partial)
        total = parts[0] + parts[1] + parts[2] + parts[3]
        x = rounded(residual.astype(jnp.float32) + norm(rounded(total), values["post"]))
        if ple:
            gate = jnp.asarray(values["gate"])
            projection = jnp.asarray(values["projection"])
            inputs = jnp.asarray(values["inputs"])[:, LAYER * WIDTH : (LAYER + 1) * WIDTH]
            gated = rounded(
                jnp.dot(x.astype(dtype), gate, preferred_element_type=jnp.float32, precision=lax.Precision.DEFAULT)
            )
            gated = rounded(jax.nn.gelu(gated, approximate=True))
            mixed = rounded(gated * inputs.astype(jnp.float32))
            projected = rounded(
                jnp.dot(
                    mixed.astype(dtype), projection, preferred_element_type=jnp.float32, precision=lax.Precision.DEFAULT
                )
            )
            x = rounded(x + norm(projected, values["ple_post"]))
            x = rounded(x * jnp.asarray(values["scale"]).astype(jnp.float32))
        return x.astype(dtype), norm(x, values["next"]).astype(dtype)

    arguments = (jnp.asarray(values["partial"]), jnp.asarray(values["residual"]))
    strict = jax.jit(compute).lower(*arguments).compile(compiler_options={"xla_allow_excess_precision": False})
    x, h = strict(*arguments)
    return np.asarray(x), np.asarray(h)


@pytest.mark.parametrize("rows", ROWS)
def test_one_shot_psum_is_the_mesh_order_sum_on_every_chip(mesh, rows) -> None:
    partial = _inputs(rows)["partial"]
    result = _run_psum(mesh, partial)
    expected = ((partial[0] + partial[1]) + partial[2]) + partial[3]
    for chip in range(4):
        np.testing.assert_array_equal(result[chip], expected)


@pytest.mark.parametrize("exchange", ["float32", "bfloat16"])
@pytest.mark.parametrize("ple", [False, True], ids=["attention", "mlp-ple-scale"])
@pytest.mark.parametrize("rows", ROWS)
def test_one_shot_psum_norm_matches_the_reference(mesh, rows, ple, exchange) -> None:
    values = _inputs(rows)
    x, h = _run_norm(mesh, values, ple=ple, exchange=exchange)
    for chip in range(1, 4):
        np.testing.assert_array_equal(x[chip], x[0])
        np.testing.assert_array_equal(h[chip], h[0])
    expected_x, expected_h = _reference(values, ple=ple, exchange=exchange)
    _assert_close_to_reference(x[0], expected_x)
    _assert_close_to_reference(h[0], expected_h)


@pytest.mark.parametrize("ple", [False, True], ids=["attention", "mlp-ple-scale"])
@pytest.mark.parametrize("rows", [16, 32])
def test_scatter_gather_merge_is_bitwise_the_one_shot_merge(mesh, rows, ple) -> None:
    values = _inputs(rows, seed=2)
    want = _run_norm(mesh, values, ple=ple)
    got = _run_norm(mesh, values, ple=ple, algorithm="scatter_gather")
    for value, expected in zip(got, want, strict=True):
        np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("ple", [False, True], ids=["attention", "mlp-ple-scale"])
def test_gather_rows_merge_is_the_sum_of_zero_padded_blocks(mesh, ple) -> None:
    # Chip r's rows placed in an array of zeros elsewhere: the one-shot sum adds exact zeros, so stacking the
    # row blocks must give the same bits.
    rows = 32
    values = _inputs(rows, seed=3)
    local = rows // 4
    blocks = values["partial"][:, :local]
    padded = np.zeros_like(values["partial"])
    for rank in range(4):
        padded[rank, rank * local : (rank + 1) * local] = blocks[rank]
    want = _run_norm(mesh, {**values, "partial": padded}, ple=ple)
    got = _run_norm(mesh, {**values, "partial": blocks}, ple=ple, algorithm="gather_rows")
    for value, expected in zip(got, want, strict=True):
        np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("ple", [False, True], ids=["attention", "mlp-ple-scale"])
def test_scatter_rows_merge_is_the_row_block_of_the_one_shot_merge(mesh, ple) -> None:
    from gemma4_posttrain_jax.pallas.ici_allreduce import PerLayerInput, one_shot_psum_norm
    from gemma4_posttrain_jax.tp_rollout import DP_AXIS, TP_AXIS

    rows, local = 32, 8
    values = _inputs(rows, seed=4)
    want_x, want_h = _run_norm(mesh, values, ple=ple)

    def body(partial, residual, post, nxt, ple_post, scale, gate, projection, inputs):
        me = jax.lax.axis_index(TP_AXIS)
        mine = lambda v: jax.lax.dynamic_slice_in_dim(v, me * local, local, axis=0)  # noqa: E731
        block = PerLayerInput(gate, projection, mine(inputs), LAYER, ple_post) if ple else None
        return one_shot_psum_norm(
            partial,
            mine(residual),
            post,
            nxt,
            eps=EPS,
            axis_name=TP_AXIS,
            other_axes=(DP_AXIS,),
            per_layer_input=block,
            scale=scale if ple else None,
            algorithm="scatter_rows",
        )

    names = ("residual", "post", "next", "ple_post", "scale", "gate", "projection", "inputs")
    run = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(TP_AXIS),) + (P(),) * len(names),
            out_specs=(P(TP_AXIS), P(TP_AXIS)),
            check_vma=False,
        )
    )
    x, h = run(jnp.asarray(values["partial"].reshape(-1, COLS)), *(jnp.asarray(values[name]) for name in names))
    # chip r holds rows r*8..r*8+7 of the one-shot result, so the stacked blocks are its full result
    np.testing.assert_array_equal(np.asarray(x), want_x[0])
    np.testing.assert_array_equal(np.asarray(h), want_h[0])


def test_norm_gather_equals_the_gather_rows_merge(mesh) -> None:
    from gemma4_posttrain_jax.pallas.ici_allreduce import norm_gather
    from gemma4_posttrain_jax.tp_rollout import DP_AXIS, TP_AXIS

    rows, local = 32, 8
    values = _inputs(rows, seed=5)
    blocks = values["partial"][:, :local]
    want_x, want_h = _run_norm(mesh, {**values, "partial": blocks}, ple=False, algorithm="gather_rows")

    def body(partial, residual, post, nxt):
        me = jax.lax.axis_index(TP_AXIS)
        mine = jax.lax.dynamic_slice_in_dim(residual, me * local, local, axis=0)
        return norm_gather(partial, mine, post, nxt, eps=EPS, axis_name=TP_AXIS, other_axes=(DP_AXIS,))

    run = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(P(TP_AXIS), P(), P(), P()), out_specs=(P(TP_AXIS), P(TP_AXIS)), check_vma=False
        )
    )
    x, h = run(
        jnp.asarray(blocks.reshape(-1, COLS)), *(jnp.asarray(values[name]) for name in ("residual", "post", "next"))
    )
    np.testing.assert_array_equal(np.asarray(x), want_x[0])
    np.testing.assert_array_equal(np.asarray(h).reshape(4, rows, COLS), want_h)


@pytest.mark.parametrize("exchange", ["float32", "bfloat16"])
@pytest.mark.parametrize("ple", [False, True], ids=["attention", "mlp-ple-scale"])
def test_rows_do_not_depend_on_the_row_count(mesh, ple, exchange) -> None:
    values = _inputs(max(ROWS), seed=1)
    base_x, base_h = _run_norm(mesh, _take_rows(values, 8), ple=ple, exchange=exchange)
    base_sum = _run_psum(mesh, values["partial"][:, :8])
    for rows in ROWS[1:]:
        x, h = _run_norm(mesh, _take_rows(values, rows), ple=ple, exchange=exchange)
        np.testing.assert_array_equal(x[:, :8], base_x)
        np.testing.assert_array_equal(h[:, :8], base_h)
        np.testing.assert_array_equal(_run_psum(mesh, values["partial"][:, :rows])[:, :8], base_sum)
