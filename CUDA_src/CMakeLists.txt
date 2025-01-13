cuda_add_executable(deviceQuery 00-deviceQuery.cpp)
cuda_add_executable(main
        main_app.cu
        csrOperations.cu
)