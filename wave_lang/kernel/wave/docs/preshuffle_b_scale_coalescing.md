# Dynamic Bounds Checks Break B-Scale Read Coalescing

## Background

MXFP4 GEMM kernels with preshuffle-B require each thread to read 16
scale bytes per K-iteration. The read coalescing pass
(`partition_strided_operators.py`) merges individual byte reads into
wider vector loads.

## Static dims: works perfectly

With static (compile-time-known) dimensions, bounds checks simplify to
`true` and are eliminated. The coalescer freely merges all 16 byte
reads into clean `vector<16xi8>` loads with no `vector.from_elements`
reassembly. The preshuffle index mapping is *not* the problem — the
physical addresses are contiguous.

Tested with `test_dbuf_4wave_mxfp_preshuffle_b_gemm` (static dims):
8 clean `vector<16xi8>` B-scale loads in the prologue, zero
`vector.from_elements`.

## Dynamic dims: coalescing fragments

With dynamic dimensions, each byte read carries a bounds check
(`index < dim`). The preshuffle index mapping produces per-byte bounds
check expressions that are symbolically distinct, even when they are
numerically equivalent at runtime. This prevents the coalescer from
merging adjacent reads that have incompatible mask expressions.

Tested with `test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm`:
the 16 byte reads fragment into 4 loads of mixed widths, then get
reassembled element-by-element:

```mlir
// 4 global loads of different widths.
%867 = vector.load ... : vector<2xi8>
%876 = vector.load ... : vector<16xi8>   // only bytes 0 and 13 used
%888 = vector.load ... : vector<8xi8>
%900 = vector.load ... : vector<4xi8>

// 16 × extract + from_elements to reassemble.
%901 = vector.extract %867[0]  : i8 from vector<2xi8>
%902 = vector.extract %867[1]  : i8 from vector<2xi8>
%903 = vector.extract %876[0]  : i8 from vector<16xi8>
%904 = vector.extract %888[0]  : i8 from vector<8xi8>
...
%916 = vector.extract %876[13] : i8 from vector<16xi8>
%917 = vector.from_elements %901, ..., %916 : vector<16xi8>
```

This repeats 8 times (once per MMA scale input).

**16 bytes needed, 30 bytes fetched.** The multiway coalescer sees two
byte reads at positions 0 and 13 in the same 16-byte aligned window and
emits a `vector<16xi8>` load — wasting 14 of 16 bytes.

## Root cause

The pairwise merge in `_pairwise_merge` and the mask builder in
`_build_wide_mask_expr` require that adjacent reads either share the
same mask condition or that conditions can be unified. The preshuffle
index mapping feeds each byte's row index through a different
expression (e.g. `floor(offset_i / (K/2)) < N`), producing per-lane
conditions that differ symbolically even though they agree numerically
under divisibility assumptions.

The `_try_collapse_uniform_conditions` pass (added in commit
`cd5e7e00`) uses numeric probing + divisibility substitutions to detect
and collapse equivalent conditions, but this only helps *after*
pairwise merging has already decided which reads can pair up. If the
pairwise merge rejects a pair due to mask incompatibility, the
condition collapse never gets a chance to run on that pair.

## Possible fixes

1. **Condition collapse before/during pairwise merge:** Move the
   numeric equivalence check earlier so that reads with provably-
   equivalent bounds checks can be merged even when their mask
   expressions differ symbolically.

2. **Shared memory for B-scale:** Route B-scale reads through LDS with
   a custom layout where each thread's 16 bytes are contiguous. This
   sidesteps the problem entirely — shared memory reads have no dynamic
   bounds checks. See the `preshuffle_scale_to_shared` pass (currently
   commented out in `compile.py`).

3. **Simplify bounds checks earlier in the pipeline:** Canonicalize the
   preshuffle index expressions under divisibility assumptions before
   the coalescing pass runs, so per-lane conditions are syntactically
   identical rather than just numerically equivalent.
