# =========================================
# File Name : CMakeLists.txt
# Encoding  : UTF-8
# =========================================

### JP: CMake のバージョン
cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "PhotonicRT")
set(TARGET_NAME ${PROJECT_NAME})

project(${PROJECT_NAME}
    VERSION 0.1
    DESCRIPTION "This is a simple project to develop a renderer"
    LANGUAGES C CXX CUDA                # JP: CUDA をリストに登録することで， FindCUDA module の呼び出しが不要
)

# -----------------------------------------------------
# ビルドの設定
# -----------------------------------------------------

### JP: C++ 標準の設定
set(CMAKE_CXX_STANDARD 17)              # JP: 使用する C++ のバージョン
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # JP: 上記の標準を必須として使用することを要求
set(CMAKE_CXX_EXTENSIONS OFF)           # JP: ベンダー拡張は OFF

set(utils_dir ${PROJECT_SOURCE_DIR}/utils)
set(
    CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    "${utils_dir}/cmake")

find_package(CUDAToolkit 12.0 REQUIRED) 
find_package(OptiX9)

if(OptiX9_FOUND)
    set(OPTIX_INCLUDE_DIR "${OPTIX9_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OptiX SDK 9.0.0 not found.")
endif()

include(${utils_dir}/cmake/BuildSettings.cmake)
include(${utils_dir}/cmake/nvcuda_compile_module.cmake)

# -----------------------------------------------------
# サブモジュールの読み込み
# -----------------------------------------------------

### utils 
add_subdirectory(${utils_dir}) # ${utils_dir} 以下にある CMakeList.txt をサブプロジェクトしてビルドに追加

### ext
set(EXT_DIR ${PROJECT_SOURCE_DIR}/ext)

### assimp
option(ASSIMP_BUILD_ZLIB "Build your own zlib" ON)
option(ASSIMP_BUILD_ASSIMP_TOOLS "If the supplementary tools for Assimp are built in addition to the library." OFF)
option(ASSIMP_BUILD_SAMPLES "If the official samples are built as well (needs Glut)." OFF)
option(ASSIMP_BUILD_TESTS "If the test suite for Assimp is built in addition to the library." OFF)
option(ASSIMP_INSTALL "Disable this if you want to use assimp as a submodule." OFF)
# Windows 環境
if(MSVC)
    option(ASSIMP_BUILD_ASSIMP_VIEW "If the Assimp view tool is built. (requires DirectX)" OFF)
endif()
add_subdirectory(${EXT_DIR}/assimp)

### DirectXTex
add_subdirectory(${EXT_DIR}/DirectXTex)

# OpenGL
find_package(OpenGL REQUIRED)

# glfw
set(OpenGL_GL_PREFERENCE LEGACY) # LEGACY - OLD, GLVND - NEW
if (WIN32)
    set(glfw_dir ${PROJECT_SOURCE_DIR}/ext/glfw)
    include_directories(${glfw_dir}/include)
    add_subdirectory(${glfw_dir} EXCLUDE_FROM_ALL)
else()
    find_package(glfw3 REQUIRED)
endif()

# imgui
set(IMGUI_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ext/imgui)
set(IMGUI_BACKENDS_DIR ${IMGUI_SOURCE_DIR}/backends)

set(IMGUI_SOURCES
    ${IMGUI_SOURCE_DIR}/examples/example_glfw_opengl3/main.cpp
    ${IMGUI_SOURCE_DIR}/imgui_demo.cpp
    ${IMGUI_SOURCE_DIR}/imgui_draw.cpp
    ${IMGUI_SOURCE_DIR}/imgui_tables.cpp
    ${IMGUI_SOURCE_DIR}/imgui.cpp
    ${IMGUI_SOURCE_DIR}/imgui_widgets.cpp
    ${IMGUI_SOURCE_DIR}/imgui.h
)

set(IMGUI_BACKEND_SOURCES
    ${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.cpp
    ${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp
    ${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.h
    ${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.h
)

add_library(imgui STATIC
    ${IMGUI_SOURCES}
    ${IMGUI_BACKEND_SOURCES}
)

target_include_directories(imgui PUBLIC
    ${IMGUI_SOURCE_DIR}
    ${IMGUI_BACKENDS_DIR}
)

target_link_libraries(imgui
    ${OPENGL_gl_LIBRARY}
    glfw
)

# json
add_subdirectory(${EXT_DIR}/json)

# openvdb
include(${utils_dir}/cmake/BuildVDB.cmake)
#add_library(nvdb_runtime STATIC .cu)
#target_link_libraries(nvdb_runtime PRIVATE nanovdb_headers)

if(OPENVDB_AVAILABLE OR WITH_OPENVDB)
    #add_library(vdb_loader STATIC src/vdbLoader.cpp)
    if(DEFINED OPENVDB_TARGET)
        #target_link_libraries(vdb_loader PRIVATE ${OPENVDB_TARGET})
    endif()
endif()

### viewer
add_subdirectory(viewer)
target_link_libraries(viewer
    PRIVATE 
    utils 
    imgui
    # ${OPENGL_gl_LIBRARY}
 )

# include_directories(${OPTIX_INCLUDE_DIR})


# -----------------------------------------------------
# .cu ファイルのコンパイル
# -----------------------------------------------------

set(CPP_VER_CUDA c++17)
set(USE_OPTIX_IR FALSE)
set(CUDA_CC 86)


### コンパイルするファイルを列挙
# ptx ファイルに変換したいファイル群
set(
    CUDA_KERNELS
    ${PROJECT_SOURCE_DIR}/kernels/toneMap.cu
    ${PROJECT_SOURCE_DIR}/kernels/copy_buffer_to_surface.cu
)

# optixir に変換したいファイル群
set(
    OPTIX_KERNELS
    ${PROJECT_SOURCE_DIR}/shaders/entry/raygen.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/ah_radiance.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/ah_shadow.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/ch_radiance.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/ch_shadow.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/miss_radiance.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/miss_shadow.cu
    ${PROJECT_SOURCE_DIR}/shaders/callable/disney_principled_brdf.cu
    ${PROJECT_SOURCE_DIR}/shaders/callable/diffuse_brdf.cu
    ${PROJECT_SOURCE_DIR}/shaders/callable/glass_bsdf.cu
    ${PROJECT_SOURCE_DIR}/shaders/callable/light_sample.cu
    ${PROJECT_SOURCE_DIR}/shaders/callable/lens.cu
)

if(OptiX9_FOUND)
    set(USE_OPTIX_IR TRUE)
else()
    set(
        ${CUDA_KERNELS}
        ${CUDA_KERNELS}
        ${OPTIX_KERNELS}
    )
    set(OPTIX_KERNELS)
endif()

# .cu ファイルをコンパイルするのに必要なヘッダファイルたち
set(
    GPU_KERNEL_DEPENDENCIES
    ${PROJECT_SOURCE_DIR}/utils/helper_math.h
    ${PROJECT_SOURCE_DIR}/utils/my_math.hpp
    ${PROJECT_SOURCE_DIR}/include/launch_params.h
    ${PROJECT_SOURCE_DIR}/shaders/params/per_ray_data.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/random_number_generator.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/shader_common.cuh
    ${PROJECT_SOURCE_DIR}/shaders/config.cuh
)

# .cu ファイルを .ptx ファイルにコンパイル (実装は nvcuda_compile_module.cmake モジュール内)
set(CUDA_NVCC_OPTIONS
    "-Xcompiler" "\"/utf-8 /Zc:__cplusplus\""   # 警告無視，標準準拠オプション
    "--generate-line-info"                      # 行番号を付与し，デバッグツール等でどの行が実行中か追跡可能にするオプション
    "--gpu-architecture=compute_${CUDA_CC}"     # GPU の構成
    "--gpu-code=sm_${CUDA_CC}" 
    "-std=${CPP_VER_CUDA}"                  
    "-cudart" "shared"
    "--use_fast_math"                           # 精度を犠牲に高速な近似計算を実行
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")   # Debug ビルドのときだけ
    list(APPEND 
        CUDA_NVCC_OPTIONS
        "-D_DEBUG=1"                    # _DEBUG を定義
        "-G"                            # GPU のデバッグ情報を含める                    
    )
endif()


nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    # 例えば build/Debug/ptxes　以下に .ptx ファイルが生成
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS ${CUDA_NVCC_OPTIONS}
)

message(STATUS "CUDA_KERNELS=${CUDA_KERNELS}")


# .cu ファイルを .optixir ファイルにコンパイル (実装は nvcuda_compile_module.cmake モジュール内) 
if(USE_OPTIX_IR)
    set(OPTIX_NVCC_OPTIONS
        "--generate-line-info"                      # 行番号を付与し，デバッグツール等でどの行が実行中か追跡可能にするオプション
        "--gpu-architecture=compute_${CUDA_CC}"     # GPU の構成
        "-std=${CPP_VER_CUDA}"
        "-cudart" "shared"
        "--use_fast_math"                           # 精度を犠牲に高速な近似計算を実行
        "--relocatable-device-code=true"
        "-I${OPTIX_INCLUDE_DIR}"
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug") # Debug ビルドのときだけ
        list(APPEND 
            OPTIX_NVCC_OPTIONS
            "-D_DEBUG=1"             # _DEBUG を定義
            "-G"                     # GPU のデバッグ情報を含める                    
            "-O0"                    # 最適化を無効化 
        )
    endif()
    
    nvcuda_compile_optix_ir(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS ${OPTIX_NVCC_OPTIONS}
)
endif()


# ビルドした結果をまとめる
set(BUILD_CUDA_KERNELS
  ${CUDA_PTXES} 
  ${OPTIX_PTXES}
)


# -----------------------------------------------------
# 実行ファイルの定義
# -----------------------------------------------------
set(SOURCE_DIR src)
set(INCLUDE_DIR include)

add_executable(${PROJECT_NAME}
    "${BUILD_CUDA_KERNELS}"
    "${SOURCE_DIR}/main.cpp"
    "${SOURCE_DIR}/renderer.cpp"
    "${SOURCE_DIR}/texture.cpp"
    "${SOURCE_DIR}/model.cpp"
    "${SOURCE_DIR}/application.cpp"
    "${INCLUDE_DIR}/optix8.hpp"
    "${INCLUDE_DIR}/renderer.hpp"
    "${INCLUDE_DIR}/texture.hpp"
    "${INCLUDE_DIR}/model.h"
    "${INCLUDE_DIR}/launch_params.h"
    "${INCLUDE_DIR}/cuda_buffer.h"
    "${INCLUDE_DIR}/ptx_data_reader.hpp"
    "${INCLUDE_DIR}/application.hpp"
    "${EXT_DIR}/stb/stb_image_write.h"
)

# -----------------------------------------------------
# ターゲットの設定
# -----------------------------------------------------
# コンパイル時のマクロ定義の追加
target_compile_definitions(
    ${PROJECT_NAME} PRIVATE
    /utf-8
    /fsanitize=address
    /O2
)

if (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/utf-8>
                      $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

# CMake 上の変数を C++ に渡す
if (WIN32)
    add_definitions(-DNOMINMAX)
endif()

if(USE_OPTIX_IR)
    add_definitions("-DUSE_OPTIX_IR")
endif()


# インクルードパスの追加
target_include_directories(
    ${PROJECT_NAME} PRIVATE
    "${OPTIX_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/${INCLUDE_DIR}"
    ${EXT_DIR}/stb
    ${EXT_DIR}/DirectXTex
    ${EXT_DIR}/assimp/include
    ${EXT_DIR}/json
)

# ライブラリやターゲットをリンク
target_link_libraries(
    ${PROJECT_NAME}
    ${OPENGL_gl_LIBRARY}
    utils
    viewer
    glfw
    imgui
    assimp
    DirectXTex
    nlohmann_json::nlohmann_json
    CUDA::cudart        # FindCudaToolkit の成果物
    CUDA::cuda_driver   # FindCudaToolkit の成果物 (cuCtxGetCurrent() はここにいる)
)


message(STATUS "----------------------------------------")
message(STATUS "Info :")
message(STATUS "Project name = ${PROJECT_NAME}")
message(STATUS "Project version = ${PROJECT_VERSION}")
message(STATUS "CMAKE_VERSION = ${CMAKE_VERSION}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_LIBDIR = ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "OPTIX = ${optix_LIBRARY}")
message(STATUS "CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")