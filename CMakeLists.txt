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
# copy data assets to output directory
# ----------------------------------------------------------------------

file(GLOB shadertoy_shaders CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/data/shaders/*.comp")
file(GLOB shadertoy_textures CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/data/textures/*")

set(shadertoy_assets_stamp "${CMAKE_CURRENT_BINARY_DIR}/.shadertoy_assets_stamp")

add_custom_command(
    OUTPUT "${shadertoy_assets_stamp}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:shadertoy>/data/shaders"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:shadertoy>/data/textures"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${shadertoy_shaders}
        "$<TARGET_FILE_DIR:shadertoy>/data/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/data/textures"
        "$<TARGET_FILE_DIR:shadertoy>/data/textures"
    COMMAND ${CMAKE_COMMAND} -E touch "${shadertoy_assets_stamp}"
    DEPENDS ${shadertoy_shaders} ${shadertoy_textures}
    COMMENT "Copying shaders and textures to output directory"
    VERBATIM
)

add_custom_target(shadertoy_assets DEPENDS "${shadertoy_assets_stamp}")
add_dependencies(shadertoy shadertoy_assets)

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------

find_package(mango REQUIRED)
target_link_libraries(shadertoy PUBLIC mango::vulkan)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build: ${CMAKE_BUILD_TYPE}")
