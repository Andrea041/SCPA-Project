cmake_minimum_required(VERSION 3.7)
project(SCPA_Project)

set(CC 75)

find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# Aggiungi i file sorgente per l'eseguibile
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
)

# Collegamento delle librerie
target_link_libraries(SCPA_Project PUBLIC OpenMP::OpenMP_C)
target_link_libraries(SCPA_Project PUBLIC cjson)

set_target_properties(SCPA_Project PROPERTIES
        INSTALL_RPATH "$ORIGIN/../cJSON/build"
)

find_path(CJSON_INCLUDE_DIR cJSON.h PATHS ../cJSON/include/cjson)
find_library(CJSON_LIBRARY cjson PATHS ../cJSON/build)

if(CJSON_INCLUDE_DIR AND CJSON_LIBRARY)
    include_directories(${CJSON_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Could not find cJSON!")
endif()

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/Common)

set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)

add_subdirectory(CUDA_src)

# Configurazione dell'utente
if($ENV{USER} STREQUAL "pierfrancesco")
    message(STATUS "Compilazione per Pierfrancesco Lijoi completata")
    add_compile_definitions(USER_PIERFRANCESCO)
elseif($ENV{USER} STREQUAL "andreaandreoli")
    message(STATUS "Compilazione per Andrea Andreoli completata")
    add_compile_definitions(USER_ANDREA)
else()
    message(STATUS "Compilazione da server di dipartimento completata")
endif()
