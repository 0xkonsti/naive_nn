cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

project(naive_nn VERSION 0.1.0 LANGUAGES C)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

file(GLOB_RECURSE SOURCES src/*.c src/*.h)
file(GLOB_RECURSE HEADERS include/*.h)

include_directories(include)

add_library(naive_nn STATIC ${SOURCES} ${HEADERS})

set(REQUIRED_LIBRARIES "")
include(CheckLibraryExists)

check_library_exists(m exp "" LIBM_EXP)
check_library_exists(m tanh "" LIBM_TANH)

if (LIBM_EXP AND LIBM_TANH)
    list(APPEND REQUIRED_LIBRARIES m)
endif ()

message(STATUS "Required libraries: ${REQUIRED_LIBRARIES}")

target_link_libraries(naive_nn ${REQUIRED_LIBRARIES})

add_executable(main test/main.c)

target_link_libraries(main naive_nn)
