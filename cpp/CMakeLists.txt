cmake_minimum_required(VERSION 3.13)
project(zipformer_asr_axera_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

set(AXERA_TARGET "AX650" CACHE STRING "AXERA target: AX650 or AX630C")
set_property(CACHE AXERA_TARGET PROPERTY STRINGS AX650 AX630C)
string(TOUPPER "${AXERA_TARGET}" AXERA_TARGET)
if(AXERA_TARGET STREQUAL "AX650")
  set(ASR_EXECUTABLE zipformer_asr_ax650)
  set(DEFAULT_MODELS_DIR "inputs/axmodels_650N")
  set(DEFAULT_BSP_MSP_DIR
      "${CMAKE_SOURCE_DIR}/toolchains/ax650n_bsp_sdk/msp/out")
elseif(AXERA_TARGET STREQUAL "AX630C")
  set(ASR_EXECUTABLE zipformer_asr_ax630c)
  set(DEFAULT_MODELS_DIR "inputs/axmodels_630C")
  set(DEFAULT_BSP_MSP_DIR
      "${CMAKE_SOURCE_DIR}/toolchains/ax620e_bsp_sdk/msp/out/arm64_glibc")
else()
  message(FATAL_ERROR "AXERA_TARGET must be AX650 or AX630C")
endif()

set(BSP_MSP_DIR "${DEFAULT_BSP_MSP_DIR}" CACHE PATH
    "BSP directory containing include/ and lib/")
if(NOT EXISTS "${BSP_MSP_DIR}/include/ax_engine_api.h")
  message(FATAL_ERROR
          "${AXERA_TARGET} BSP not found: ${BSP_MSP_DIR}. "
          "Set BSP_MSP_DIR to the directory containing include/ and lib/.")
endif()

set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(KALDI_NATIVE_FBANK_ARCHIVE
    "${CMAKE_SOURCE_DIR}/third_party/kaldi-native-fbank-1.22.3.tar.gz")
set(KISSFFT_ARCHIVE
    "${CMAKE_SOURCE_DIR}/third_party/kissfft-febd4caeed32e33ad8b2e0bb5ea77542c40f18ec.zip")
if(NOT EXISTS "${KALDI_NATIVE_FBANK_ARCHIVE}" OR
   NOT EXISTS "${KISSFFT_ARCHIVE}")
  message(FATAL_ERROR "Missing cpp/third_party fbank archives")
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/third_party")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xzf "${KALDI_NATIVE_FBANK_ARCHIVE}"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/third_party"
  RESULT_VARIABLE KNF_EXTRACT_RESULT
)
if(NOT KNF_EXTRACT_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to extract kaldi-native-fbank archive")
endif()
set(KNF_SOURCE_DIR
    "${CMAKE_BINARY_DIR}/third_party/kaldi-native-fbank-1.22.3")
file(COPY "${KISSFFT_ARCHIVE}" DESTINATION "${CMAKE_BINARY_DIR}")
add_subdirectory("${KNF_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/knf"
                 EXCLUDE_FROM_ALL)

add_executable(${ASR_EXECUTABLE}
  zipformer_asr_ax.cpp
  src/engine_wrapper.cpp
)
target_include_directories(${ASR_EXECUTABLE} PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src
  ${KNF_SOURCE_DIR}
  ${BSP_MSP_DIR}/include
)
target_link_directories(${ASR_EXECUTABLE} PRIVATE ${BSP_MSP_DIR}/lib)
target_compile_definitions(${ASR_EXECUTABLE} PRIVATE
  AXERA_TARGET_NAME="${AXERA_TARGET}"
  DEFAULT_MODELS_DIR="${DEFAULT_MODELS_DIR}"
)
target_compile_options(${ASR_EXECUTABLE} PRIVATE
  -O3 -Wall -Wextra -Wpedantic
  -fvisibility=hidden -fdata-sections -ffunction-sections
)
target_link_libraries(${ASR_EXECUTABLE} PRIVATE
  kaldi-native-fbank-core ax_sys ax_engine ax_interpreter pthread dl m
)
target_link_options(${ASR_EXECUTABLE} PRIVATE
  -Wl,--enable-new-dtags -Wl,--allow-shlib-undefined
)
set_target_properties(${ASR_EXECUTABLE} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)
