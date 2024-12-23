cmake_minimum_required(VERSION 3.16)
project(SCPA_Project LANGUAGES C)

# Trova la libreria OpenMP
find_package(OpenMP REQUIRED)

# Specifica lo standard C
set(CMAKE_C_STANDARD 99)

# Aggiungi l'eseguibile
add_executable(SCPA_Project
        prova.c
)

# Collega OpenMP all'eseguibile
if(OpenMP_C_FOUND)
    target_link_libraries(SCPA_Project PUBLIC OpenMP::OpenMP_C)
endif()