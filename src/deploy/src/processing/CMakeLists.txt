# Minimal processing submodule target

add_library(processing_core STATIC
    core/MemoryPool.cpp
    core/PipelineRunner.cpp
    transforms/Resize.cpp
)

target_include_directories(processing_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# keep target lightweight: no extra dependencies now.
set_target_properties(processing_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)