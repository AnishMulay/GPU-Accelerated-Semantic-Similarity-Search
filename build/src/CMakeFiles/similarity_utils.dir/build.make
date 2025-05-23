# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/anish/miniconda/lib/python3.12/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/anish/miniconda/lib/python3.12/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build

# Include any dependencies generated for this target.
include src/CMakeFiles/similarity_utils.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/similarity_utils.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/similarity_utils.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/similarity_utils.dir/flags.make

src/CMakeFiles/similarity_utils.dir/codegen:
.PHONY : src/CMakeFiles/similarity_utils.dir/codegen

src/CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.o: src/CMakeFiles/similarity_utils.dir/flags.make
src/CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.o: /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/data_loader.cpp
src/CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.o: src/CMakeFiles/similarity_utils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.o"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.o -MF CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.o.d -o CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.o -c /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/data_loader.cpp

src/CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.i"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/data_loader.cpp > CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.i

src/CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.s"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/data_loader.cpp -o CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.s

src/CMakeFiles/similarity_utils.dir/utils/metrics.cpp.o: src/CMakeFiles/similarity_utils.dir/flags.make
src/CMakeFiles/similarity_utils.dir/utils/metrics.cpp.o: /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/metrics.cpp
src/CMakeFiles/similarity_utils.dir/utils/metrics.cpp.o: src/CMakeFiles/similarity_utils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/similarity_utils.dir/utils/metrics.cpp.o"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/similarity_utils.dir/utils/metrics.cpp.o -MF CMakeFiles/similarity_utils.dir/utils/metrics.cpp.o.d -o CMakeFiles/similarity_utils.dir/utils/metrics.cpp.o -c /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/metrics.cpp

src/CMakeFiles/similarity_utils.dir/utils/metrics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/similarity_utils.dir/utils/metrics.cpp.i"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/metrics.cpp > CMakeFiles/similarity_utils.dir/utils/metrics.cpp.i

src/CMakeFiles/similarity_utils.dir/utils/metrics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/similarity_utils.dir/utils/metrics.cpp.s"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/metrics.cpp -o CMakeFiles/similarity_utils.dir/utils/metrics.cpp.s

src/CMakeFiles/similarity_utils.dir/utils/timer.cpp.o: src/CMakeFiles/similarity_utils.dir/flags.make
src/CMakeFiles/similarity_utils.dir/utils/timer.cpp.o: /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/timer.cpp
src/CMakeFiles/similarity_utils.dir/utils/timer.cpp.o: src/CMakeFiles/similarity_utils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/similarity_utils.dir/utils/timer.cpp.o"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/similarity_utils.dir/utils/timer.cpp.o -MF CMakeFiles/similarity_utils.dir/utils/timer.cpp.o.d -o CMakeFiles/similarity_utils.dir/utils/timer.cpp.o -c /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/timer.cpp

src/CMakeFiles/similarity_utils.dir/utils/timer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/similarity_utils.dir/utils/timer.cpp.i"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/timer.cpp > CMakeFiles/similarity_utils.dir/utils/timer.cpp.i

src/CMakeFiles/similarity_utils.dir/utils/timer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/similarity_utils.dir/utils/timer.cpp.s"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/timer.cpp -o CMakeFiles/similarity_utils.dir/utils/timer.cpp.s

src/CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.o: src/CMakeFiles/similarity_utils.dir/flags.make
src/CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.o: /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/vector_data.cpp
src/CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.o: src/CMakeFiles/similarity_utils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.o"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.o -MF CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.o.d -o CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.o -c /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/vector_data.cpp

src/CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.i"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/vector_data.cpp > CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.i

src/CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.s"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/vector_data.cpp -o CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.s

src/CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.o: src/CMakeFiles/similarity_utils.dir/flags.make
src/CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.o: /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/benchmark.cpp
src/CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.o: src/CMakeFiles/similarity_utils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.o"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.o -MF CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.o.d -o CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.o -c /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/benchmark.cpp

src/CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.i"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/benchmark.cpp > CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.i

src/CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.s"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src/utils/benchmark.cpp -o CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.s

# Object files for target similarity_utils
similarity_utils_OBJECTS = \
"CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.o" \
"CMakeFiles/similarity_utils.dir/utils/metrics.cpp.o" \
"CMakeFiles/similarity_utils.dir/utils/timer.cpp.o" \
"CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.o" \
"CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.o"

# External object files for target similarity_utils
similarity_utils_EXTERNAL_OBJECTS =

src/libsimilarity_utils.a: src/CMakeFiles/similarity_utils.dir/utils/data_loader.cpp.o
src/libsimilarity_utils.a: src/CMakeFiles/similarity_utils.dir/utils/metrics.cpp.o
src/libsimilarity_utils.a: src/CMakeFiles/similarity_utils.dir/utils/timer.cpp.o
src/libsimilarity_utils.a: src/CMakeFiles/similarity_utils.dir/utils/vector_data.cpp.o
src/libsimilarity_utils.a: src/CMakeFiles/similarity_utils.dir/utils/benchmark.cpp.o
src/libsimilarity_utils.a: src/CMakeFiles/similarity_utils.dir/build.make
src/libsimilarity_utils.a: src/CMakeFiles/similarity_utils.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library libsimilarity_utils.a"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && $(CMAKE_COMMAND) -P CMakeFiles/similarity_utils.dir/cmake_clean_target.cmake
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/similarity_utils.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/similarity_utils.dir/build: src/libsimilarity_utils.a
.PHONY : src/CMakeFiles/similarity_utils.dir/build

src/CMakeFiles/similarity_utils.dir/clean:
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src && $(CMAKE_COMMAND) -P CMakeFiles/similarity_utils.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/similarity_utils.dir/clean

src/CMakeFiles/similarity_utils.dir/depend:
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/src /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/src/CMakeFiles/similarity_utils.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/similarity_utils.dir/depend

