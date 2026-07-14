cmake_minimum_required(VERSION 3.19)
project(shadertoy)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------
# source files
# ----------------------------------------------------------------------

set(sources
    source/shadertoy.cpp
)

# ----------------------------------------------------------------------
# build executable
# ----------------------------------------------------------------------

add_executable(shadertoy WIN32 ${sources})

# ----------------------------------------------------------------------
# copy shaders to output directory
# ----------------------------------------------------------------------

file(GLOB shadertoy_shaders CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/data/shaders/*.comp")

add_custom_command(TARGET shadertoy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:shadertoy>/data/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${shadertoy_shaders}
        "$<TARGET_FILE_DIR:shadertoy>/data/shaders"
    COMMENT "Copying shaders to output directory"
    VERBATIM
)

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------

find_package(mango REQUIRED)
target_link_libraries(shadertoy PUBLIC mango::vulkan)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build: ${CMAKE_BUILD_TYPE}")
