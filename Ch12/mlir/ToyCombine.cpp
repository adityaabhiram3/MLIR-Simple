//===- simCombine.cpp - sim High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the sim dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "toy/Dialect.h"
#include <numeric>
using namespace mlir;
using namespace sim;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // end anonymous namespace

/// Fold simple cast operations that return the same type as the input.
OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  return mlir::impl::foldCastOp(*this);
}

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every sim.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
                 FoldConstantReshapeOptPattern>(context);
}
