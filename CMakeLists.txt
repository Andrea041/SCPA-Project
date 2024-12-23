cmake_minimum_required(VERSION 3.30)
project(SCPA_Project C)

set(CMAKE_C_STANDARD 11)

# Trova MPI
find_package(MPI REQUIRED)

# Aggiungi i flag di compilazione di MPI
include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}")

# Aggiungi l'eseguibile
add_executable(SCPA_Project
        prova.c)

# Collega MPI all'eseguibile
target_link_libraries(SCPA_Project ${MPI_LIBRARIES})