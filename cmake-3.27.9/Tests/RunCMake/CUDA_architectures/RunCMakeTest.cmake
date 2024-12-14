include(RunCMake)

run_cmake(architectures-all)
run_cmake(architectures-all-major)
run_cmake(architectures-native)
run_cmake(architectures-empty)
run_cmake(architectures-invalid)

run_cmake(architectures-not-set)
include("${RunCMake_BINARY_DIR}/architectures-not-set-build/info.cmake" OPTIONAL)
message(STATUS "  CMAKE_CUDA_COMPILER_ID='${CMAKE_CUDA_COMPILER_ID}'")
message(STATUS "  CMAKE_CUDA_COMPILER_VERSION='${CMAKE_CUDA_COMPILER_VERSION}'")
message(STATUS "  CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'")

if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang" AND CMAKE_CUDA_ARCHITECTURES)
  list(GET CMAKE_CUDA_ARCHITECTURES 0 arch)
  set(CMAKE_CUDA_FLAGS --cuda-gpu-arch=sm_${arch})
  message(STATUS "Adding CMAKE_CUDA_FLAGS='${CMAKE_CUDA_FLAGS}' for CMAKE_CUDA_ARCHITECTURES=OFF with Clang.")
  set(RunCMake_TEST_OPTIONS "-DCMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")
endif()
run_cmake(architectures-off)
unset(RunCMake_TEST_OPTIONS)

if(CMAKE_CUDA_ARCHITECTURES MATCHES "([0-9]+)")
  set(arch "${CMAKE_MATCH_1}")
  run_cmake_with_options(architectures-suffix -Darch=${arch})
endif()
