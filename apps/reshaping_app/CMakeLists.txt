cmake_minimum_required(VERSION 3.9)
project(ReshapingAPP)

set(GLM_ROOT_DIR "${DEPENDENCIES_BASE_DIR}/glm")
list(APPEND CMAKE_PREFIX_PATH "${GLM_ROOT_DIR}/cmake/")
find_package(glm REQUIRED)
add_compile_definitions(FMT_USE_CHAR8_T=0)

# include extra application dependencies here
include(cli11)
include(spdlog)
include(glad REQUIRED)
include(glfw REQUIRED)
include(glm REQUIRED)
include(imgui REQUIRED)
include(imguizmo REQUIRED)
find_package(OpenGL REQUIRED)

find_library(OpenGL_LIBRARY OpenGL)

file(GLOB APP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(reshaping_app)
target_sources(reshaping_app PRIVATE ${APP_SOURCES})
target_include_directories(reshaping_app PUBLIC "${BASE_LIBS_INCLUDE_DIR}"
    "${GLM_INCLUDE_DIRS}")
target_link_libraries(reshaping_app PUBLIC
    mesh_reshaping_lib
    glfw::glfw
    ${OpenGL_LIBRARY}
    imgui::imgui
    imguizmo::imguizmo
    Eigen3::Eigen
    CLI11::CLI11
    lagrange::core
    lagrange::io
    spdlog::spdlog
    glad::glad
)

target_compile_definitions(reshaping_demo
    PRIVATE
        FMT_USE_CHAR8_T=0
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
