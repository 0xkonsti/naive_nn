
BUILD_DIR = out
SRC_DIR = src
INC_DIR = include

CMAKE_FLAGS = 

CMAKE_FLAGS_DEBUG = $(CMAKE_FLAGS)
CMAKE_FLAGS_DEBUG += -DCMAKE_BUILD_TYPE=Debug
CMAKE_FLAGS_DEBUG += -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

CMAKE_FLAGS_RELEASE = $(CMAKE_FLAGS)
CMAKE_FLAGS_RELEASE += -DCMAKE_BUILD_TYPE=Release


# Utility targets

.PHONY: build-dir
build-dir:
	@mkdir -p $(BUILD_DIR)

.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)

.PHONY: ctags
ctags:
	@ctags -R --c++-kinds=+p --fields=+iaS --extras=+q $(SRC_DIR) $(INC_DIR)

.PHONY: compile
compile:
	@cd $(BUILD_DIR) && make

.PHONY: format
format:
	@find $(SRC_DIR) $(INC_DIR) -name "*.c" -o -name "*.h" | xargs clang-format -i

.PHONY: run-debug
run-debug: debug
	@cd $(BUILD_DIR) && ./main

.PHONY: run-release
run-release: release
	@cd $(BUILD_DIR) && ./main

# Build the project\

.PHONY: cmake-debug
cmake-debug: build-dir
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS_DEBUG) ..

.PHONY: cmake-release
cmake-release: build-dir
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS_RELEASE) ..

.PHONY: debug
debug: cmake-debug
	@cd $(BUILD_DIR) && make

.PHONY: release
release: cmake-release
	@cd $(BUILD_DIR) && make
