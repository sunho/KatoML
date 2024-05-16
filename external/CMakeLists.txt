if (ENABLE_MLIR)
  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
  set(LLVM_BUILD_EXAMPLES OFF CACHE STRING "")
  set(LLVM_TARGETS_TO_BUILD "X86;NVPTX;AMDGPU" CACHE STRING "")
#   set(LLVM_TARGETS_TO_BUILD "X86;NVPTX;AMDGPU" CACHE STRING "")
# DLLVM_ENABLE_PROJECTS=mlir \
#    -DLLVM_BUILD_EXAMPLES=ON \
#    -DLLVM_TARGETS_TO_BUILD="" \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DLLVM_ENABLE_ASSERTIONS=ON
#   add_subdirectory(llvm-project)
endif()

if (ENABLE_TESTS)
  add_subdirectory(Catch2)
  include(Catch)
endif()

add_subdirectory(pybind11)

