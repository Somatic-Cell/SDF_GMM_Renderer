# =========================================
# File Name : FindOptiX8.cmake
# Encoding  : UTF-8
# =========================================
# OptiX ライブラリを読み込むために，頑張って optix.h　を検出する

# OptiX SDK がどこにインストールされているかを探す
# プロジェクトの一つ上の階層に OptiX があるかどうか探す（GUI などから設定できるようにもする）
if (WIN32) 
    # Windows 環境だったら． 
    # WIN32 は Windows 環境だったら True を返すことに注意．　
    # 32bit OS でも 64bit OS でも True を返す 
    set(OPTIX8_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0" CACHE PATH "Path to OptiX installed location.")
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  if(WIN32)
    message(SEND_ERROR "ERROR: Make sure when selecting the generator, you select one with Win64 or x64.")
  endif()
  message(FATAL_ERROR "ERROR: OptiX only supports builds configured for 64 bits.")
endif()

#message(STATUS "OptiX include path = ${OPTIX8_INSTALL_DIR}/include")
#message(STATUS "Windows!! ${OPTIX8_INSTALL_DIR}/include")

# optix.h の検出
# まず，ユーザ指定の OPTIX8_INSTALL_DIR 配下の include/ に optix.h があるかどうかを探す
find_path(
    OPTIX8_INCLUDE_DIR 
    NAMES optix.h
    PATHS "${OPTIX8_INSTALL_DIR}/include"
    NO_DEFAULT_PATH                 # システム標準のパスは検索しない
)

# 標準パスも探してみる
find_path(
    OPTIX8_INCLUDE_DIR 
    NAMES optix.h 
)

# message(STATUS "OptiX include path = ${OPTIX8_INCLUDE_DIR}")

# エラーメッセージ
include(FindPackageHandleStandardArgs)  # 標準で搭載されている便利なモジュール：https://cmake.org/cmake/help/latest/module/FindPackageHandleStandardArgs.html
# パッケージが見つかった場合は，<パッケージ名>_FOUND 変数に TRUE/FALSE を自動的に設定
find_package_handle_standard_args(
    OptiX8                  # パッケージ名
    DEFAULT_MSG             # エラーメッセージの表示方法
    OPTIX8_INCLUDE_DIR      # 判定に使う変数
)
mark_as_advanced(OPTIX8_INCLUDE_DIR) # GUI 上には表示しない