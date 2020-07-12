include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

if(CMAKE_CUDA_COMPILER)
        try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/has_cuda_gpu.cu
        COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
        RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
    message(STATUS ${RUN_OUTPUT_VAR})

    if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
        set(HAVE_CUDA_H ON)
        enable_language(CUDA)
        message(STATUS "CUDA Enabled")
        set(CUDA_FOUND TRUE)
    else()
        message(STATUS "CUDA Disabled")
        set(HAVE_CUDA_H OFF)
        set(CUDA_FOUND FALSE)
    endif()
endif()