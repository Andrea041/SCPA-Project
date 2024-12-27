cmake_minimum_required(VERSION 3.16)
project(SCPA_Project C)

# Specifica lo standard C
set(CMAKE_C_STANDARD 99)

# Trova la libreria OpenMP
find_package(OpenMP REQUIRED)

# Aggiungi l'eseguibile
add_executable(SCPA_Project
        # File sorgenti
        src/main_app.c
        src/functionsIO.c
        src/csrTool.c
        src/csrOperations.c
        src/hll_ellpack_Tool.c

        # File di intestazione
        libs/matrixLists.h
        libs/csrTool.h
        libs/csrOperations.h
        libs/hll_ellpack_Tool.h
        libs/data_structure.h
)

if(OpenMP_C_FOUND)
    target_link_libraries(SCPA_Project PUBLIC OpenMP::OpenMP_C)
else()
    message(WARNING "OpenMP not found!")
endif()

# Rilevamento dell'utente che utilizza il programma di calcolo
if($ENV{USER} STREQUAL "pierfrancesco")
    message(STATUS "Compilazione per Pierfrancesco Lijoi")
    add_compile_definitions(USER_PIERFRANCESCO)
elseif($ENV{USER} STREQUAL "andreaandreoli")
    message(STATUS "Compilazione per Andrea Andreoli")
    add_compile_definitions(USER_ANDREA)
else()
    message(WARNING "Utente non riconosciuto")
endif()