cmake_minimum_required(VERSION 3.7)
project(SCPA_Project)

# Architettura CUDA
set(CC 75)

# Configurazione dei moduli di CMake
set(CMAKE_MODULE_PATH /data/aandreoli/cJSON/build ${CMAKE_MODULE_PATH})

# Trova le librerie richieste
find_package(cJSON REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# Aggiungi i file sorgente per l'eseguibile principale
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

# Collegamento delle librerie per il progetto principale
target_link_libraries(SCPA_Project PUBLIC OpenMP::OpenMP_C)
target_link_libraries(SCPA_Project PUBLIC cJSON::cJSON)

# Includi directory aggiuntive
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/data/aandreoli/cuda-samples/Common)

# Imposta i flag di compilazione
set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)

# Includi la directory CUDA_src
add_subdirectory(CUDA_src)

# Configurazione per utenti specifici
if($ENV{USER} STREQUAL "pierfrancesco")
    message(STATUS "Compilazione per Pierfrancesco Lijoi completata")
    add_compile_definitions(USER_PIERFRANCESCO)
elseif($ENV{USER} STREQUAL "andreaandreoli")
    message(STATUS "Compilazione per Andrea Andreoli completata")
    add_compile_definitions(USER_ANDREA)
else()
    message(STATUS "Compilazione da server di dipartimento completata")
endif()