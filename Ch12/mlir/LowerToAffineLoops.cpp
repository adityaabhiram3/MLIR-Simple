//====- LowerToAffineLoops.cpp - Partial lowering from sim to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of sim operations to a combination of
// affine loops and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"
#include "toy/Passes.h"
#include <iostream>
#include <cstdlib>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include <limits.h>
#include <float.h>
using namespace mlir;

//===----------------------------------------------------------------------===//
// simToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc.getOperation()->getBlock();
  alloc.getOperation()->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as sim functions have no control flow.
  auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
  dealloc.getOperation()->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input a rewriter, an array of memRefOperands corresponding
/// to the operands of the input operation, and the set of loop induction
/// variables for the iteration. It returns a value to store at the current
/// index of the iteration.
using LoopIterationFn = function_ref<Value(PatternRewriter &rewriter,
                                           ArrayRef<Value> memRefOperands,
                                           ArrayRef<Value> loopIvs)>;

static void lowerOpToLoops(Operation *op, ArrayRef<Value> operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();
  llvm::errs() << " \n\n operands : " << operands[0] << "\n";
  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create an empty affine loop for each of the dimensions within the shape.
  SmallVector<Value, 4> loopIvs;
  for (auto dim : tensorType.getShape()) {
    auto loop = rewriter.create<AffineForOp>(loc, /*lb=*/0, dim, /*step=*/1);
    loopIvs.push_back(loop.getInductionVar());

    // Update the rewriter insertion point to the beginning of the loop.
    rewriter.setInsertionPointToStart(loop.getBody());
  }
  //std::cout << "\n\n loop ivs : " << loopIvs[0] << "\n";

  // Generate a call to the processing function with the rewriter, the memref
  // operands, and the loop induction variables. This function will return the
  // value to store at the current index.
  Value valueToStore = processIteration(rewriter, operands, loopIvs);
  rewriter.create<AffineStoreOp>(loc, valueToStore, alloc,
                                 llvm::makeArrayRef(loopIvs));

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// simToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](PatternRewriter &rewriter, ArrayRef<Value> memRefOperands,
              ArrayRef<Value> loopIvs) {
          // Generate an adaptor for the remapped operands of the BinaryOp. This
          // allows for using the nice named accessors that are generated by the
          // ODS.
          typename BinaryOp::OperandAdaptor binaryAdaptor(memRefOperands);

          // Generate loads for the element of 'lhs' and 'rhs' at the inner
          // loop.
          auto loadedLhs =
              rewriter.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
          auto loadedRhs =
              rewriter.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

          // Create the binary operation performed on the loaded values.
          return rewriter.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
        });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<sim::AddOp, AddFOp>;
using MulOpLowering = BinaryOpLowering<sim::MulOp, MulFOp>;
using SubOpLowering = BinaryOpLowering<sim::SubOp, SubFOp>;

//===----------------------------------------------------------------------===//
// simToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<sim::ConstantOp> {
  using OpRewritePattern<sim::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sim::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.value();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
              0, *std::max_element(valueShape.begin(), valueShape.end())))
       constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    }
    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.getValues<FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// simToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<sim::ReturnOp> {
  using OpRewritePattern<sim::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sim::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "sim.return" directly to "std.return".
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SimToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(sim::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](PatternRewriter &rewriter, ArrayRef<Value> memRefOperands,
              ArrayRef<Value> loopIvs) {
          // Generate an adaptor for the remapped operands of the TransposeOp.
          // This allows for using the nice named accessors that are generated
          // by the ODS.
          sim::TransposeOpOperandAdaptor transposeAdaptor(memRefOperands);
          Value input = transposeAdaptor.input();

          // Transpose the elements by generating a load from the reverse
          // indices.
          SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return rewriter.create<AffineLoadOp>(loc, input, reverseIvs);
        });
    return success();
  }
};


//===----------------------------------------------------------------------===//
// simToAffine RewritePatterns: Sum operations
//===----------------------------------------------------------------------===//


static void lowerOpToLoopsSum(Operation *op, ArrayRef<Value> operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {

  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();
  auto rand = (*op->operand_type_begin()).cast<TensorType>();

  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  Value zero = rewriter.create<ConstantIndexOp>(loc,0);

  /*
  //llvm::errs() << "\n\n integer : " << integer << "\n";
  SmallVector<Value, 4> Index0;
  //llvm::errs() << "\n\n shape";
  for(int i = rand.getRank(); i > 0; i--) {
      Index0.push_back(rewriter.create<ConstantIndexOp>(loc,0));
  }
  */
  //Value valueToStoreInit = processIteration(rewriter, operands, Index0);
  //auto initzero = rewriter.create<SubFOp>(loc, valueToStoreInit, valueToStoreInit);
  auto initzero = rewriter.getFloatAttr(rewriter.getF64Type(), 0);
  auto constinit = rewriter.create<ConstantOp>(loc, initzero);
  rewriter.create<StoreOp>(loc, constinit, alloc, zero);

  SmallVector<Value, 4> loopIvs;
  for (auto dim : rand.getShape()) {
    auto loop = rewriter.create<AffineForOp>(loc, 0, dim, 1);
    loopIvs.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToStart(loop.getBody());
  }



  Value valueToStore = processIteration(rewriter, operands, loopIvs);

  //Value index = rewriter.create<SubIOp>(loc, loopIvs[0],loops[0] );

  Value temp = rewriter.create<AffineLoadOp>(loc, alloc, zero);
  Value final = rewriter.create<AddFOp>(loc, temp, valueToStore);

  rewriter.create<AffineStoreOp>(loc, final, alloc, zero);
  rewriter.replaceOp(op, alloc);

}

struct SumOpLowering : public ConversionPattern {
  SumOpLowering(MLIRContext *ctx)
      : ConversionPattern(sim::SumOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoopsSum(
        op, operands, rewriter,
        [loc](PatternRewriter &rewriter, ArrayRef<Value> memRefOperands,
              ArrayRef<Value> loopIvs) {
          sim::SumOpOperandAdaptor SumAdaptor(memRefOperands);
          Value input = SumAdaptor.input();
          return rewriter.create<AffineLoadOp>(loc, input, loopIvs);
    });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// simToAffine RewritePatterns: Max operations
//===----------------------------------------------------------------------===//


static void lowerOpToLoopsMax(Operation *op, ArrayRef<Value> operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {

  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();
  auto rand = (*op->operand_type_begin()).cast<TensorType>();

  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  Value zero = rewriter.create<ConstantIndexOp>(loc,0);

  /*
  SmallVector<Value, 4> Index0;
  //llvm::errs() << "\n\n shape";
  for(int i = rand.getRank(); i > 0; i--) {
      Index0.push_back(rewriter.create<ConstantIndexOp>(loc,0));
  }
  */
  //Value valueToStoreInit = processIteration(rewriter, operands, Index0);
  auto initzero = rewriter.getFloatAttr(rewriter.getF64Type(), FLT_MIN);
  auto constinit = rewriter.create<ConstantOp>(loc, initzero);
  rewriter.create<AffineStoreOp>(loc, constinit, alloc, zero);


  SmallVector<Value, 4> loopIvs;
  SmallVector<Value, 4> loops;
  for (auto dim : rand.getShape()) {
    auto loop = rewriter.create<AffineForOp>(loc, 0, dim, 1);
    loopIvs.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  Value valueToStore = processIteration(rewriter, operands, loopIvs);

  Value temp = rewriter.create<AffineLoadOp>(loc, alloc, zero);
  Value final = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT ,valueToStore, temp);
  Value out = rewriter.create<SelectOp>(loc, final, temp, valueToStore);

  rewriter.create<AffineStoreOp>(loc, out, alloc, zero);
  rewriter.replaceOp(op, alloc);
}


struct MaxOpLowering : public ConversionPattern {
  MaxOpLowering(MLIRContext *ctx)
      : ConversionPattern(sim::MaxOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoopsMax(
        op, operands, rewriter,
        [loc](PatternRewriter &rewriter, ArrayRef<Value> memRefOperands,
              ArrayRef<Value> loopIvs) {

          sim::MaxOpOperandAdaptor MaxAdaptor(memRefOperands);
          Value input = MaxAdaptor.input();

          return rewriter.create<AffineLoadOp>(loc, input, loopIvs);
    });
    return success();
  }
};




//===----------------------------------------------------------------------===//
// simToAffine RewritePatterns: Min operations
//===----------------------------------------------------------------------===//


static void lowerOpToLoopsMin(Operation *op, ArrayRef<Value> operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {

  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto rand = (*op->operand_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();

  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  Value zero = rewriter.create<ConstantIndexOp>(loc, 0);

  /*
  SmallVector<Value, 4> Index0;
  //llvm::errs() << "\n\n shape";
  for(int i = rand.getRank(); i > 0; i--) {
      Index0.push_back(rewriter.create<ConstantIndexOp>(loc,0));
  }
  */
  //Value valueToStoreInit = processIteration(rewriter, operands, Index0);
  auto initzero = rewriter.getFloatAttr(rewriter.getF64Type(), FLT_MAX);
  auto constinit = rewriter.create<ConstantOp>(loc, initzero);
  rewriter.create<AffineStoreOp>(loc, constinit, alloc, zero);


  SmallVector<Value, 4> loopIvs;
  for (auto dim : rand.getShape()) {
    auto loop = rewriter.create<AffineForOp>(loc, 0, dim, 1);
    loopIvs.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  Value valueToStore = processIteration(rewriter, operands, loopIvs);



  Value temp = rewriter.create<AffineLoadOp>(loc, alloc, zero);
  Value final = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT ,valueToStore, temp);
  Value out = rewriter.create<SelectOp>(loc,final, valueToStore, temp);

  rewriter.create<AffineStoreOp>(loc, out, alloc, zero);
  rewriter.replaceOp(op, alloc);
}

struct MinOpLowering : public ConversionPattern {
  MinOpLowering(MLIRContext *ctx)
      : ConversionPattern(sim::MinOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoopsMin(
        op, operands, rewriter,
        [loc](PatternRewriter &rewriter, ArrayRef<Value> memRefOperands,
              ArrayRef<Value> loopIvs) {

          sim::MinOpOperandAdaptor MinAdaptor(memRefOperands);
          Value input = MinAdaptor.input();

          return rewriter.create<AffineLoadOp>(loc, input, loopIvs);
    });
    return success();
  }
};


static void lowerOpToLoopsConv(Operation *op, ArrayRef<Value> operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {

}

struct ConvOpLowering : public ConversionPattern {
  ConvOpLowering(MLIRContext *ctx)
      : ConversionPattern(sim::ConvOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();

    sim::ConvOpOperandAdaptor ConvAdaptor(operands);
    Value input = ConvAdaptor.input();
    Value filter = ConvAdaptor.filter();

    auto iter = op->operand_type_begin();
    auto inputtype = (*iter).cast<TensorType>();
    iter++;
    auto filtertype = (*iter).cast<TensorType>();

    auto initzero = rewriter.getFloatAttr(rewriter.getF64Type(), 0);
    auto constinit = rewriter.create<ConstantOp>(loc, initzero);


    llvm::errs() <<"\n\n input shape : " <<inputtype.getShape()[0] << " " << inputtype.getShape()[1] << "\n" ;
    llvm::errs() <<"\n\n filter shape : " <<filtertype.getShape()[0] << " " << filtertype.getShape()[1] << "\n" ;

    auto resulttype = (*op->result_type_begin()).cast<TensorType>();
    auto memRefType = convertTensorToMemRef(resulttype);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<Value, 4> OuterLoop;
    for (auto dim : resulttype.getShape()) {
      auto loop = rewriter.create<AffineForOp>(loc, 0, dim, 1);
      OuterLoop.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    auto storeop = rewriter.create<AffineStoreOp>(loc, constinit ,alloc, OuterLoop);
    //auto *parentBlockone = storeop.getOperation()->getBlock();


    SmallVector<Value, 4> FilterLoop;
    for (auto dim : filtertype.getShape()) {
      auto loop = rewriter.create<AffineForOp>(loc, 0, dim, 1);
      FilterLoop.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    auto loadop1 = rewriter.create<AffineLoadOp>(loc, filter, FilterLoop);
    SmallVector<Value, 2> ImageIdx;
    auto x = rewriter.create<AddIOp>(loc, FilterLoop[0], OuterLoop[0]);
    auto y = rewriter.create<AddIOp>(loc, FilterLoop[1], OuterLoop[1]);
    ImageIdx.push_back(x);
    ImageIdx.push_back(y);
    auto loadop2 = rewriter.create<LoadOp>(loc, input, ImageIdx);
    auto res =  rewriter.create<MulFOp>(loc, loadop1, loadop2);
    auto valadd = rewriter.create<AffineLoadOp>(loc, alloc, OuterLoop);
    auto add =  rewriter.create<AddFOp>(loc, res, valadd );
    rewriter.create<AffineStoreOp>(loc, add, alloc, OuterLoop);

    //storeop.getOperation()->moveBefore(&parentBlockone->back());
    rewriter.replaceOp(op, alloc);
    return success();
  }
};




} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// simToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the sim operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the sim dialect.
namespace {
struct SimToAffineLoweringPass
    : public PassWrapper<SimToAffineLoweringPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void SimToAffineLoweringPass::runOnFunction() {
  auto function = getFunction();

  // We only lower the main function as we expect that all other functions have
  // been inlined.
  if (function.getName() != "main")
    return;

  // Verify that the given main has no inputs and results.
  if (function.getNumArguments() || function.getType().getNumResults()) {
    function.emitError("expected 'main' to have 0 inputs and 0 results");
    return signalPassFailure();
  }

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  target.addLegalDialect<AffineDialect, StandardOpsDialect>();

  // We also define the Sim dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Sim operations that don't want
  // to lower, `Sim.print`, as `legal`.
  target.addIllegalDialect<sim::SimDialect>();
  target.addLegalOp<sim::PrintOp>();
  //target.addLegalOp<sim::SumOp>();


  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the sim operations.
  OwningRewritePatternList patterns;
  patterns.insert<AddOpLowering, ConstantOpLowering, MulOpLowering,
                  ReturnOpLowering, TransposeOpLowering, SubOpLowering, SumOpLowering, MaxOpLowering, MinOpLowering, ConvOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the sim IR (e.g. matmul).
std::unique_ptr<Pass> mlir::sim::createLowerToAffinePass() {
  return std::make_unique<SimToAffineLoweringPass>();
}
