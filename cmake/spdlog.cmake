#
# Copyright 2020 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#
if(TARGET spdlog::spdlog)
    return()
endif()

message(STATUS "Third-party (external): creating target 'spdlog::spdlog'")

include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    # GIT_TAG        v1.6.1
    GIT_TAG        v1.15.2
)

option(SPDLOG_INSTALL "Generate the install target" ON)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "spdlog")
FetchContent_MakeAvailable(spdlog)

set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(spdlog PROPERTIES FOLDER third_party)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" OR
   "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(spdlog PRIVATE
        "-Wno-sign-conversion"
    )
endif()
