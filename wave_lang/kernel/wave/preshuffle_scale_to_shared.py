# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Pass to remap preshuffle scale shared memory layout for contiguous reads.

After minimize_global_loads, preshuffle scale data has:
  - 1 merged global read (ept=4) with preshuffle mapping
  - 1 merged shared write (ept=4) at {K: k_expr :1, M: m_expr :4}
  - 8 shared reads (ept=1) at scattered (K, M) positions

The MMA access pattern reads 8 scale bytes per thread at 2 K-group pairs
x 4 M-tile positions, which are scattered in the natural [K_groups, M]
shared layout. This pass remaps to a custom [M, 8] layout where each
thread's 8 reads become contiguous in the last dimension, enabling
merge_contiguous_reads to produce vector<8xi8> loads.

See preshuffle_scale_to_shared_design.md for the full design.
"""

import logging

import sympy

from .._support.indexing import IndexSequence
from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import Allocate, ExtractSlice, Read, Write, get_custom
from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
)
from .utils.general_utils import (
    has_write_shared_user,
)
from .utils.symbol_utils import subs_idxc

logger = logging.getLogger(__name__)


def _is_preshuffle_scale_read(node) -> bool:
    """Detect a global Read with a preshuffle mapping (non-identity,
    no dynamic values) that feeds a shared Write."""
    custom = get_custom(node)
    if not isinstance(custom, Read):
        return False
    if subs_idxc(custom.memory_type.address_space) != GLOBAL_ADDRESS_SPACE:
        return False
    if custom.mapping is None or custom.mapping.is_input_identity():
        return False
    # Exclude actual gathers with dynamic values.
    if len(custom.mapping_dynamic_vals) > 0:
        return False
    if not has_write_shared_user(custom):
        return False
    return True


def _layout_row(k, m):
    """Compute row index in the custom [M, 8] shared layout.

    row = T0 (the reading thread), encoding which thread reads this byte.
    """
    return (k % 4) * 16 + (m % 16) + (m // 64) * 64


def _layout_col(k, m):
    """Compute column index in the custom [M, 8] shared layout.

    col = byte offset within the thread's 8-byte read vector.
    """
    return ((m % 64) // 16) * 2 + k // 4


def preshuffle_scale_to_shared(trace: CapturedTrace, constraints: list[Constraint]):
    """Remap preshuffle scale shared memory to a custom layout where
    each thread's 8 scale reads are contiguous."""
    matched_reads = trace.walk(_is_preshuffle_scale_read)
    if not matched_reads:
        return

    hardware_constraint = next(
        c for c in constraints if isinstance(c, HardwareConstraint)
    )

    # Group by memory to process each shared allocation once.
    processed_memories = set()

    for read_node in matched_reads:
        read = get_custom(read_node)
        if read.memory in processed_memories:
            continue
        processed_memories.add(read.memory)

        # Find all global reads for this memory.
        all_global_reads = [
            get_custom(n) for n in matched_reads if get_custom(n).memory == read.memory
        ]

        # Find the shared allocation through the first global read's
        # shared write user.
        alloc_resized = False
        for global_read in all_global_reads:
            # Flag global read so gather_to_shared skips it.
            global_read.fx_node.meta["skip_gather_to_shared"] = True

            # Find the shared write fed by this global read.
            for user in global_read.users:
                if not isinstance(user, Write):
                    continue
                if subs_idxc(user.memory_type.address_space) != SHARED_ADDRESS_SPACE:
                    continue

                shared_write = user

                # Resize the shared allocation once per memory.
                if not alloc_resized:
                    allocate = get_custom(shared_write.memory)
                    assert isinstance(allocate, Allocate)
                    old_distributed = list(allocate.distributed_shape)
                    m_size = old_distributed[-1] - allocate.padding
                    new_distributed = (m_size, 8)
                    logger.info(
                        f"preshuffle_scale_to_shared: resizing "
                        f"{old_distributed} -> {new_distributed}"
                    )
                    allocate.update_arg("distributed_shape", new_distributed)
                    allocate.update_arg("padding", 0)
                    alloc_resized = True

                dims = list(shared_write.index.keys())
                k_dim, m_dim = dims[0], dims[1]
                k_expr = shared_write.index[k_dim].start
                m_expr = shared_write.index[m_dim].start
                write_ept = subs_idxc(shared_write.elements_per_thread)

                # Collect shared reads before modifying the graph.
                shared_reads = []
                for write_user_node in list(shared_write.fx_node.users):
                    sr = get_custom(write_user_node)
                    if isinstance(sr, Read) and (
                        subs_idxc(sr.memory_type.address_space) == SHARED_ADDRESS_SPACE
                    ):
                        shared_reads.append(sr)

                # Split write (ept=N) into N individual writes (ept=1)
                # at layout-remapped positions.
                new_writes = []
                with shared_write.graph.inserting_before(shared_write.fx_node):
                    for i in range(write_ept):
                        m_i = m_expr + i
                        row = _layout_row(k_expr, m_i)
                        col = _layout_col(k_expr, m_i)

                        extract = ExtractSlice(
                            shared_write.register_, [i], [1], [1]
                        ).add_to_graph(shared_write.graph, loc=shared_write.location)

                        new_write = Write(
                            extract,
                            shared_write.memory,
                            elements_per_thread=1,
                        ).add_to_graph(shared_write.graph, loc=shared_write.location)
                        new_write.index = {
                            k_dim: IndexSequence(row, 1, 1),
                            m_dim: IndexSequence(col, 1, 1),
                        }
                        new_write.vector_shapes = shared_write.vector_shapes
                        new_writes.append(new_write)

                # Update write dependencies on shared reads, then remap
                # their indices.
                new_write_nodes = [
                    w if isinstance(w, type(shared_write.fx_node)) else w
                    for w in new_writes
                ]

                # Compute the common row from the first read. All reads
                # for the same thread map to the same row by construction
                # (verified numerically in the design doc), but sympy
                # cannot prove this across different (k, m) pairs. Using
                # a single row expression lets merge_contiguous_reads see
                # all 8 reads as contiguous in the column dimension.
                first_sr = shared_reads[0]
                k_first = first_sr.index[k_dim].start
                m_first = first_sr.index[m_dim].start
                common_row = _layout_row(k_first, m_first)

                seen_cols = set()
                for sr in shared_reads:
                    # Replace old write with new writes in dependency list.
                    old_deps = list(sr._write_dependency)
                    new_deps = [d for d in old_deps if d != shared_write.fx_node]
                    new_deps.extend(new_write_nodes)
                    sr.update_arg("_write_dependency", new_deps)

                    # The col value is thread-independent (same for all
                    # threads, differs only between expansion copies).
                    # Evaluate to a concrete integer so merge_contiguous_reads
                    # can detect contiguity.
                    k_r = sr.index[k_dim].start
                    m_r = sr.index[m_dim].start
                    col_expr = _layout_col(k_r, m_r)
                    col = int(
                        sympy.sympify(col_expr).subs(
                            {s: 0 for s in sympy.sympify(col_expr).free_symbols}
                        )
                    )
                    assert 0 <= col < 8, f"Unexpected col value {col}"
                    assert col not in seen_cols, f"Duplicate col {col}"
                    seen_cols.add(col)

                    sr.index = {
                        k_dim: IndexSequence(common_row, 1, 1),
                        m_dim: IndexSequence(col, 1, 1),
                    }
                    # Data is already in the correct register layout
                    # after reading from remapped positions.
                    sr.update_arg("mapping", None)

                # Erase old write (no more users after dependency update).
                shared_write.graph.erase_node(shared_write.fx_node)
