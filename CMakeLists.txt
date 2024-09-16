cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

project(naive_nn VERSION 0.1.0 LANGUAGES C)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

file(GLOB_RECURSE SOURCES src/*.c src/*.h)
file(GLOB_RECURSE HEADERS include/*.h)

include_directories(include)

add_library(naive_nn STATIC ${SOURCES} ${HEADERS})

add_executable(main test/main.c)

target_link_libraries(main naive_nn)
