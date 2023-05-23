cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME "Squares")
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

# Specify all include directories
include_directories(
        "${CMAKE_CURRENT_LIST_DIR}/libs/JSON/include"
        "${CMAKE_CURRENT_LIST_DIR}/libs"
)

# Specify paths to find SFML and TGUI
set(SFML_DIR "${CMAKE_CURRENT_LIST_DIR}/libs/SFML/lib/cmake/SFML")
set(TGUI_DIR "${CMAKE_CURRENT_LIST_DIR}/libs/TGUI/lib/cmake/TGUI")

# Copy required DLL files
file(GLOB ALL_REQUIRED_DLL "libs/required_dlls/*.dll")
file(COPY ${ALL_REQUIRED_DLL} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(GLOB TGUI_DLL "libs/TGUI/bin/*.dll")
file(COPY ${TGUI_DLL} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Source files to be used in the project
set(SOURCE_FILES
        src/Game.cpp
        src/CoordinateEntity.cpp
        src/CoordinateEntity.hpp
        src/MapGenerator.cpp
        src/MapGenerator.h
        # Add additional source files here
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link libraries
find_package(SFML 2.5.1 COMPONENTS system window graphics network audio REQUIRED)
find_package(TGUI 0.9 REQUIRED)
target_link_libraries(${PROJECT_NAME} sfml-audio sfml-graphics sfml-window sfml-system tgui)

link_libraries(stdc++fs) # Link the filesystem
