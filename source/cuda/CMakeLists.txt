if (CUDA_FOUND)
    add_library(ariadne-cuda OBJECT
        cuda_lib.cu
    )
    target_compile_options(ariadne-cuda PRIVATE
        "SHELL:-gencode arch=compute_52,code=sm_52"
        "SHELL:-gencode arch=compute_60,code=sm_60"
        "SHELL:-gencode arch=compute_61,code=sm_61"
        "SHELL:-gencode arch=compute_62,code=sm_62"
        "SHELL:-gencode arch=compute_70,code=sm_70"
        "SHELL:-gencode arch=compute_72,code=sm_72"
        "SHELL:-gencode arch=compute_75,code=sm_75"
        "SHELL:-gencode arch=compute_80,code=sm_80"
    )
endif()