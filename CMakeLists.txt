cmake_minimum_required(VERSION 3.7)
project(SCPA_Project)

set(CC 75)

execute_process(
        COMMAND whoami
        OUTPUT_VARIABLE CURRENT_USER
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(CURRENT_USER STREQUAL "aandreoli")
    set(CMAKE_MODULE_PATH /data/aandreoli/cJSON/build ${CMAKE_MODULE_PATH})
    add_compile_definitions(USER_ANDREA)
    include_directories(${CUDA_INCLUDE_DIRS} /data/aandreoli/cuda-samples/Common)
elseif(CURRENT_USER STREQUAL "plijoi")
    set(CMAKE_MODULE_PATH /data/plijoi/cJSON/build ${CMAKE_MODULE_PATH})
    add_compile_definitions(USER_PIERFRANCESCO)
    include_directories(${CUDA_INCLUDE_DIRS} /data/plijoi/cuda-samples/Common)
endif()

find_package(cJSON REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(SCPA_Project
        src/main_app.c
        src/functionsIO.c
        src/csrTool.c
        src/csrOperations.c
        src/hll_ellpack_Tool.c
        src/hll_Operations.c

        libs/matrixLists.h
        libs/csrTool.h
        libs/csrOperations.h
        libs/hll_ellpack_Tool.h
        libs/data_structure.h
        libs/costants.h
        libs/hll_Operations.h
        libs/mmio.h
)

target_link_libraries(SCPA_Project PUBLIC OpenMP::OpenMP_C)
target_link_libraries(SCPA_Project PUBLIC cJSON::cJSON)

set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)

add_subdirectory(CUDA_src)
