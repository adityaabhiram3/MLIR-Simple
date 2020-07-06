# Toy-Language-Extension

Instructions To Run

build/bin/toyc-ch12 "filename".sim -emit=mlir
build/bin/toyc-ch12 "filename".sim -emit=mlir-affine
build/bin/toyc-ch12 "filename".sim -emit=mlir-llvm

build/bin/mlir-cpu-runner "mlir-llvm-code-genrated-above".llvm -O3 -e main -entry-point-result=void

