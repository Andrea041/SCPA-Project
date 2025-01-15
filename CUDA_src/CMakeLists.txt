cuda_add_executable(deviceQuery 00-deviceQuery.cpp)
cuda_add_executable(SCPA-Project-CUDA
        main_app_cuda.cu
        csrOperations.cu
        csrTool.cu

        ../src/functionsIO.c
        ../../cJSON/cJSON.c
)