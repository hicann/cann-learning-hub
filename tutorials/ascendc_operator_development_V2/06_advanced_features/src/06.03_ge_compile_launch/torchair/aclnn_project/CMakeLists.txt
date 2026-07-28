cmake_minimum_required(VERSION 3.16)
project(ge_torchair_aclnn LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0
            ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

separate_arguments(TORCH_CXX_FLAGS_LIST UNIX_COMMAND "${TORCH_CXX_FLAGS}")

add_library(ge_torchair_aclnn_ops SHARED aclnn_add_pytorch.cpp)
target_compile_options(ge_torchair_aclnn_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS_LIST}>
)
target_include_directories(ge_torchair_aclnn_ops PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(ge_torchair_aclnn_ops PRIVATE
    -Wl,--no-as-needed
    Python3::Python
    ${TORCH_LIBRARIES}
    -Wl,--as-needed
)
