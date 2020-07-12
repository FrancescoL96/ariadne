if (CUDA_FOUND)
    add_library(ariadne-cuda OBJECT
        cuda_lib.cu
    )
endif()