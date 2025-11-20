# Centralized RKNN find / defaults for local testing
# Allow overriding from command line or top-level CMake

set(TARGET_SOC "rk3588" CACHE STRING "Target SoC for RKNN (override with -DTARGET_SOC=...)")


set(CMAKE_SYSTEM_NAME "Linux" CACHE STRING "CMake system name for RKNN path construction")


set(TARGET_LIB_ARCH "aarch64" CACHE STRING "Target library architecture (aarch64/armhf)")


# Use a dedicated variable pointing to the project's third_party folder so we don't rely on CMAKE_CURRENT_SOURCE_DIR mutation.
set(RKNN_THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party" CACHE PATH "Path to project's third_party directory")


# rknn runtime
# for rknpu2
if (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk3576" OR TARGET_SOC STREQUAL "rk356x" OR TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103" OR TARGET_SOC STREQUAL "rv1126b")
    set(RKNN_PATH ${RKNN_THIRD_PARTY_DIR}/rknpu2)
    if (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk356x" OR TARGET_SOC STREQUAL "rk3576")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknnrt.so)
    endif()
    if (TARGET_SOC STREQUAL "rv1126b")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknnrt.so)
    endif()
    if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103")
        set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/armhf-uclibc/librknnmrt.so)
    endif()
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include PARENT_SCOPE)
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)

endif()

# for rknpu1
if(TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    set(RKNN_PATH ${RKNN_THIRD_PARTY_DIR}/rknpu1)

    set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/librknn_api.so)

    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include PARENT_SCOPE)
endif()
install(PROGRAMS ${LIBRKNNRT} DESTINATION lib)
set(LIBRKNNRT ${LIBRKNNRT} PARENT_SCOPE)


message(STATUS "find_rknn: TARGET_SOC=${TARGET_SOC}")
message(STATUS "find_rknn: RKNN_THIRD_PARTY_DIR=${RKNN_THIRD_PARTY_DIR}")
message(STATUS "find_rknn: CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}")
message(STATUS "find_rknn: TARGET_LIB_ARCH=${TARGET_LIB_ARCH}")
message(STATUS "find_rknn: RKNN_PATH=${RKNN_PATH}")
message(STATUS "find_rknn: LIBRKNNRT=${LIBRKNNRT}")
message(STATUS "find_rknn: LIBRKNNRT_INCLUDES=${LIBRKNNRT_INCLUDES}")