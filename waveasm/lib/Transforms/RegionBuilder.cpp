// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Transforms/RegionBuilder.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "waveasm/Dialect/WaveASMOps.h"

using namespace mlir;
using namespace waveasm;

/// Check if a value is a memref in LDS (workgroup address space).
/// Used to detect memref iter_args that carry LDS buffer offsets.
static bool isLDSMemRefValue(Value val) {
  auto memrefType = dyn_cast<MemRefType>(val.getType());
  if (!memrefType)
    return false;
  return isLDSMemRef(memrefType);
}

/// Resolve the LDS base offset for a memref value.
/// Returns the SGPR value holding the byte offset, or nullptr.
static Value resolveLDSOffset(Value memref, TranslationContext &ctx,
                              OpBuilder &builder, Location loc) {
  // First check if we have a tracked LDS base offset
  if (auto baseOffset = ctx.getLDSBaseOffset(memref)) {
    Value offsetVal = *baseOffset;
    // If the offset is an immediate, materialize it as an SGPR
    if (isa<ImmType>(offsetVal.getType())) {
      auto sregType = ctx.createSRegType();
      return S_MOV_B32::create(builder, loc, sregType, offsetVal);
    }
    // If already an SGPR, use it directly
    if (isa<SRegType>(offsetVal.getType()) ||
        isa<PSRegType>(offsetVal.getType())) {
      return offsetVal;
    }
    // Otherwise, it's a VGPR - convert to SGPR via readfirstlane
    auto sregType = ctx.createSRegType();
    return V_READFIRSTLANE_B32::create(builder, loc, sregType, offsetVal);
  }
  // No tracked LDS base offset for this memref. Return nullptr so the caller
  // can report a proper error rather than silently emitting offset 0 (which
  // would produce a kernel that reads from the wrong LDS address).
  return nullptr;
}

/// Check if an scf.for region iter arg is used exclusively as a scaled MFMA
/// accumulator (destC operand). Used to decide whether the init arg
/// should be an AGPR on gfx950+ targets.
///
/// Only matches ScaledMFMAOp -- regular MFMAOp uses VGPR accumulators in
/// the C++ backend, so AGPR init would cause a type mismatch.
static bool isAccumulatorIterArg(scf::ForOp forOp, unsigned iterArgIdx) {
  auto regionIterArgs = forOp.getRegionIterArgs();
  if (iterArgIdx >= regionIterArgs.size())
    return false;
  Value regionArg = regionIterArgs[iterArgIdx];

  for (OpOperand &use : regionArg.getUses()) {
    Operation *user = use.getOwner();
    // Only scaled MFMA uses AGPR accumulators in this backend
    if (!isa<amdgpu::ScaledMFMAOp>(user))
      return false;
    if (use.getOperandNumber() != 2)
      return false;
  }
  return !regionArg.use_empty();
}

// Forward declaration
LogicalResult translateOperation(Operation *op, TranslationContext &ctx);

LogicalResult RegionBuilder::translateOp(Operation *op) {
  return translateOperation(op, ctx);
}

LoopOp RegionBuilder::buildLoopFromSCFFor(scf::ForOp forOp) {
  auto &builder = ctx.getBuilder();
  auto loc = forOp.getLoc();

  // Collect loop-carried values: [induction_var, iter_args...]
  SmallVector<Value> initArgs;
  auto lowerBound = ctx.getMapper().getMapped(forOp.getLowerBound());
  if (!lowerBound) {
    forOp.emitError("lower bound not mapped");
    return nullptr;
  }

  // Convert lower bound to sreg if it's an immediate (loop counter needs sreg
  // type)
  Value lowerBoundValue = *lowerBound;
  if (isa<ImmType>(lowerBoundValue.getType())) {
    auto sregType = ctx.createSRegType();
    lowerBoundValue =
        S_MOV_B32::create(builder, loc, sregType, lowerBoundValue);
  }
  initArgs.push_back(lowerBoundValue);

  // Track which iter_arg indices (0-based, relative to iter_args not including
  // induction var) are memref iter_args. We need this to propagate LDS offsets
  // to block arguments and to handle yield correctly.
  SmallVector<bool> isMemrefIterArg;

  for (auto [iterIdx, arg] : llvm::enumerate(forOp.getInitArgs())) {
    bool isLDSMemref = isLDSMemRefValue(arg);
    isMemrefIterArg.push_back(isLDSMemref);

    if (isLDSMemref) {
      // Memref iter_arg: resolve its LDS base offset and carry it as an SGPR.
      // This implements the ping-pong double-buffering pattern where each
      // iteration swaps which LDS buffer is "current" vs "next".
      Value offsetSgpr = resolveLDSOffset(arg, ctx, builder, loc);
      if (!offsetSgpr) {
        forOp.emitError("LDS memref iter_arg has no tracked base offset");
        return nullptr;
      }
      initArgs.push_back(offsetSgpr);
    } else if (auto mapped = ctx.getMapper().getMapped(arg)) {
      Value mappedValue = *mapped;
      // Convert immediate iter args to register types
      if (isa<ImmType>(mappedValue.getType())) {
        // Infer register size from the original SCF type.
        // For vector types (e.g. vector<4xf32> for MFMA accumulators),
        // we need a register with matching element count and alignment.
        int64_t regSize = 1;
        if (auto vecType = dyn_cast<VectorType>(arg.getType())) {
          regSize = vecType.getNumElements();
        }
        int64_t regAlign = regSize > 1 ? regSize : 1;

        // On gfx950, use AGPRs for MFMA accumulator iter_args.
        // This keeps accumulators in the upper half of the unified register
        // file (a0, a1, ...) so that VGPR indices stay within the 256
        // hardware limit. Without AGPRs, large tiles would need v256+ which
        // the assembler rejects.
        bool useAGPR = llvm::isa<GFX950TargetAttr>(ctx.getTarget());
        if (useAGPR && isAccumulatorIterArg(forOp, iterIdx)) {
          auto aregType = ctx.createARegType(regSize, regAlign);
          mappedValue = V_MOV_B32::create(builder, loc, aregType, mappedValue);
        } else {
          auto vregType = ctx.createVRegType(regSize, regAlign);
          mappedValue = V_MOV_B32::create(builder, loc, vregType, mappedValue);
        }
      }
      initArgs.push_back(mappedValue);
    } else {
      forOp.emitError("init arg not mapped");
      return nullptr;
    }
  }

  // Create loop op (result types inferred from initArgs)
  auto loopOp = LoopOp::create(builder, loc, initArgs);
  Block &bodyBlock = loopOp.getBodyBlock();

  // Map loop induction variable and iter args to block arguments
  ctx.getMapper().mapValue(forOp.getInductionVar(), bodyBlock.getArgument(0));

  size_t argIdx = 1;
  size_t iterIdx = 0;
  for (Value origArg : forOp.getRegionIterArgs()) {
    if (argIdx < bodyBlock.getNumArguments()) {
      Value blockArg = bodyBlock.getArgument(argIdx++);
      ctx.getMapper().mapValue(origArg, blockArg);

      // For memref iter_args, propagate the LDS base offset to the block
      // argument. The block argument is an SGPR carrying the LDS byte offset,
      // and we need vector.load/store/gather_to_lds handlers to be able to
      // look up this offset when they encounter the memref.
      //
      // INVARIANT: LDS base offsets are keyed on the *original SCF Values*
      // (not the mapped waveasm Values). Handlers look up the offset using
      // the unmapped SCF key (origArg), which the value mapper maps to the
      // waveasm block arg. If a remapping step is inserted between
      // RegionBuilder and the handlers, this association will break.
      if (iterIdx < isMemrefIterArg.size() && isMemrefIterArg[iterIdx]) {
        // The block argument itself IS the LDS offset (as an SGPR).
        // Set it as the LDS base offset for the original memref SSA value
        // so that operations inside the loop body can find it.
        ctx.setLDSBaseOffset(origArg, blockArg);
      }
      iterIdx++;
    }
  }

  // Translate loop body
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (failed(translateOp(&op))) {
      forOp.emitError("failed to translate loop body operation");
      return nullptr;
    }
  }

  // Build loop increment and condition
  Value inductionVar = bodyBlock.getArgument(0);

  auto step = ctx.getMapper().getMapped(forOp.getStep());
  auto upperBound = ctx.getMapper().getMapped(forOp.getUpperBound());

  if (!step || !upperBound) {
    forOp.emitError("step or upper bound not mapped");
    return nullptr;
  }

  auto sregType = ctx.createSRegType();
  Value nextIV =
      S_ADD_U32::create(builder, loc, sregType, sregType, inductionVar, *step)
          .getDst();
  // S_CMP only accepts SGPR operands. If upper bound is a VGPR (e.g. from
  // v_min_i32 in split-K trip count computation), convert to SGPR first.
  Value ub = *upperBound;
  if (isVGPRType(ub.getType())) {
    ub = V_READFIRSTLANE_B32::create(builder, loc, sregType, ub);
  }
  Value cond = S_CMP_LT_U32::create(builder, loc, sregType, nextIV, ub);

  // Collect iter args from yield
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> iterArgs;
  iterArgs.push_back(nextIV);

  size_t yieldIdx = 0;
  for (Value result : yieldOp.getResults()) {
    bool isMemref =
        yieldIdx < isMemrefIterArg.size() && isMemrefIterArg[yieldIdx];
    yieldIdx++;

    if (isMemref) {
      // For memref yield values, resolve the LDS base offset of the yielded
      // memref. This is the key to the double-buffer swap: the yield swaps
      // which memref is passed to which iter_arg position, so the SGPR
      // carrying the LDS offset also swaps.
      Value offsetSgpr = resolveLDSOffset(result, ctx, builder, loc);
      if (!offsetSgpr) {
        yieldOp.emitError("yielded LDS memref has no tracked base offset");
        return nullptr;
      }
      iterArgs.push_back(offsetSgpr);
    } else if (auto mapped = ctx.getMapper().getMapped(result)) {
      iterArgs.push_back(*mapped);
    } else {
      yieldOp.emitError("yield result not mapped");
      return nullptr;
    }
  }

  // Create condition terminator
  ConditionOp::create(builder, loc, cond, iterArgs);

  // Map loop results to for results.
  // scf.for results correspond 1:1 with iter_args (no induction variable).
  // waveasm.loop results include the induction variable at index 0,
  // then iter_args starting at index 1.
  assert(forOp.getResults().size() + 1 == loopOp.getResults().size() &&
         "result count mismatch between scf.for and waveasm.loop");
  size_t resIdx = 0;
  for (Value forRes : forOp.getResults()) {
    Value loopResult = loopOp.getResults()[resIdx + 1];
    ctx.getMapper().mapValue(forRes, loopResult);

    // For memref iter_args, propagate the LDS base offset to the loop result.
    // After the loop, the epilogue may use these results in vector.load/store
    // operations that need to know the LDS base offset. The loop result SGPR
    // carries the final (post-swap) LDS offset, so we set it as the LDS base
    // offset for the corresponding scf.for result memref.
    //
    // Index alignment: resIdx corresponds to iter_arg index because scf.for
    // results exclude the induction variable (same as isMemrefIterArg
    // indexing).
    if (resIdx < isMemrefIterArg.size() && isMemrefIterArg[resIdx]) {
      ctx.setLDSBaseOffset(forRes, loopResult);
    }
    resIdx++;
  }

  return loopOp;
}

IfOp RegionBuilder::buildIfFromSCFIf(scf::IfOp ifOp) {
  auto &builder = ctx.getBuilder();
  auto loc = ifOp.getLoc();

  // Get mapped condition
  auto condition = ctx.getMapper().getMapped(ifOp.getCondition());
  if (!condition) {
    ifOp.emitError("condition not mapped");
    return nullptr;
  }

  // Convert condition to SGPR if needed.
  // Immediates come from arith.cmpi mapping results to immediate placeholders.
  // VGPRs come from arith.select chains (e.g. workgroup reordering Piecewise
  // expressions) where all operands are uniform but the select handler
  // materialises through V_CNDMASK_B32.  Reading lane 0 is safe because the
  // condition is uniform across the wavefront.
  Value conditionValue = *condition;
  if (isa<ImmType>(conditionValue.getType())) {
    auto sregType = ctx.createSRegType();
    conditionValue = S_MOV_B32::create(builder, loc, sregType, conditionValue);
  } else if (isVGPRType(conditionValue.getType())) {
    auto sregType = ctx.createSRegType();
    conditionValue =
        V_READFIRSTLANE_B32::create(builder, loc, sregType, conditionValue);
  }

  // Collect mapped yield values from the SCF regions so we can reconcile
  // their types before creating the waveasm.if.  Both branches must yield
  // compatible register kinds (e.g. both SGPR or both VGPR).
  bool hasElse = !ifOp.getElseRegion().empty();

  // Helper: translate a region body and collect its mapped yield values.
  auto translateRegionYields =
      [&](Region &scfRegion, Block &wasmBlock,
          SmallVectorImpl<Value> &yieldVals) -> LogicalResult {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&wasmBlock);

    for (Operation &op : scfRegion.front().without_terminator())
      if (failed(translateOp(&op)))
        return failure();

    auto scfYield = cast<scf::YieldOp>(scfRegion.front().getTerminator());
    for (Value res : scfYield.getResults()) {
      auto mapped = ctx.getMapper().getMapped(res);
      if (!mapped) {
        scfYield.emitError("yield result not mapped");
        return failure();
      }
      yieldVals.push_back(*mapped);
    }
    return success();
  };

  // Use placeholder result types; we fix them up after translating both
  // branches and reconciling yield types.
  auto scfThenYield =
      cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
  SmallVector<Type> resultTypes(scfThenYield.getNumOperands(),
                                ctx.createVRegType());
  auto waveIfOp =
      IfOp::create(builder, loc, resultTypes, conditionValue, hasElse);

  // Translate then region.
  SmallVector<Value> thenYieldVals;
  if (failed(translateRegionYields(ifOp.getThenRegion(),
                                   waveIfOp.getThenBlock(), thenYieldVals))) {
    ifOp.emitError("failed to translate then region");
    return nullptr;
  }

  // Translate else region.
  SmallVector<Value> elseYieldVals;
  if (hasElse) {
    if (failed(translateRegionYields(
            ifOp.getElseRegion(), *waveIfOp.getElseBlock(), elseYieldVals))) {
      ifOp.emitError("failed to translate else region");
      return nullptr;
    }
  }

  // Reconcile yield types so both branches produce compatible register kinds.
  // The then branch typically yields ARegs (MFMA accumulators) or VGPRs while
  // the else branch yields immediates (zero constants for the OOB/skip path).
  // Promote each mismatched value to match the other branch's type.
  if (hasElse) {
    // Helper: convert |val| to |targetType|, inserting at the end of |block|.
    // V_MOV_B32 accepts Imm/VReg sources and can produce VReg or AReg output.
    auto promoteToType = [&](Value &val, Type targetType, Block &block) {
      if (typesCompatible(val.getType(), targetType))
        return;
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&block);
      if (isa<ImmType>(val.getType())) {
        // Imm -> VReg/AReg via V_MOV_B32.
        val = V_MOV_B32::create(builder, loc, targetType, val);
      } else if (isSGPRType(val.getType())) {
        // SGPR -> VReg via V_ADD_U32(zero, sgpr), then AReg if needed.
        auto vregType = ctx.createVRegType();
        auto zeroImm = ctx.createImmType(0);
        auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);
        Value zeroVgpr = V_MOV_B32::create(builder, loc, vregType, zeroConst);
        val = V_ADD_U32::create(builder, loc, vregType, zeroVgpr, val);
        if (isAGPRType(targetType))
          val = V_MOV_B32::create(builder, loc, targetType, val);
      } else if (isVGPRType(val.getType()) && isAGPRType(targetType)) {
        val = V_MOV_B32::create(builder, loc, targetType, val);
      } else if (isAGPRType(val.getType()) && isVGPRType(targetType)) {
        val = V_MOV_B32::create(builder, loc, targetType, val);
      }
    };

    for (auto [thenVal, elseVal] : llvm::zip(thenYieldVals, elseYieldVals)) {
      if (typesCompatible(thenVal.getType(), elseVal.getType()))
        continue;
      // Use the then-branch type as the target since it carries the
      // computation result (MFMAs, etc.).
      Type target = thenVal.getType();
      promoteToType(elseVal, target, *waveIfOp.getElseBlock());
      promoteToType(thenVal, target, waveIfOp.getThenBlock());
    }
  }

  // Determine final result types from the (now-reconciled) then yields.
  for (auto [i, val] : llvm::enumerate(thenYieldVals))
    waveIfOp.getResult(i).setType(val.getType());

  // Create yield ops in each branch.
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&waveIfOp.getThenBlock());
    YieldOp::create(builder, loc, thenYieldVals);
  }
  if (hasElse) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(waveIfOp.getElseBlock());
    YieldOp::create(builder, loc, elseYieldVals);
  }

  // Map if results.
  for (auto [ifRes, waveRes] :
       llvm::zip(ifOp.getResults(), waveIfOp.getResults()))
    ctx.getMapper().mapValue(ifRes, waveRes);

  return waveIfOp;
}
