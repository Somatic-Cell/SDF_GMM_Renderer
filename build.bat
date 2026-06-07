rmdir /S build
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/Users/sy415/workspace/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release --verbose
cd ..