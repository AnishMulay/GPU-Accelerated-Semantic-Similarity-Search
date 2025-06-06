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
include tests/CMakeFiles/test_brute_force_comparison.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test_brute_force_comparison.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_brute_force_comparison.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_brute_force_comparison.dir/flags.make

tests/CMakeFiles/test_brute_force_comparison.dir/codegen:
.PHONY : tests/CMakeFiles/test_brute_force_comparison.dir/codegen

tests/CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.o: tests/CMakeFiles/test_brute_force_comparison.dir/flags.make
tests/CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.o: /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/tests/test_brute_force_comparison.cpp
tests/CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.o: tests/CMakeFiles/test_brute_force_comparison.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.o"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.o -MF CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.o.d -o CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.o -c /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/tests/test_brute_force_comparison.cpp

tests/CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.i"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/tests/test_brute_force_comparison.cpp > CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.i

tests/CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.s"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/tests/test_brute_force_comparison.cpp -o CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.s

# Object files for target test_brute_force_comparison
test_brute_force_comparison_OBJECTS = \
"CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.o"

# External object files for target test_brute_force_comparison
test_brute_force_comparison_EXTERNAL_OBJECTS =

tests/test_brute_force_comparison: tests/CMakeFiles/test_brute_force_comparison.dir/test_brute_force_comparison.cpp.o
tests/test_brute_force_comparison: tests/CMakeFiles/test_brute_force_comparison.dir/build.make
tests/test_brute_force_comparison: tests/CMakeFiles/test_brute_force_comparison.dir/compiler_depend.ts
tests/test_brute_force_comparison: src/libsimilarity_utils.a
tests/test_brute_force_comparison: src/libcpu_implementations.a
tests/test_brute_force_comparison: /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so
tests/test_brute_force_comparison: /usr/lib/x86_64-linux-gnu/libpthread.a
tests/test_brute_force_comparison: /usr/local/lib/libfaiss_avx2.so
tests/test_brute_force_comparison: /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so
tests/test_brute_force_comparison: /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so
tests/test_brute_force_comparison: /usr/lib/x86_64-linux-gnu/libmkl_core.so
tests/test_brute_force_comparison: /usr/lib/x86_64-linux-gnu/libiomp5.so
tests/test_brute_force_comparison: /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so
tests/test_brute_force_comparison: /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so
tests/test_brute_force_comparison: /usr/lib/x86_64-linux-gnu/libmkl_core.so
tests/test_brute_force_comparison: /usr/lib/x86_64-linux-gnu/libiomp5.so
tests/test_brute_force_comparison: tests/CMakeFiles/test_brute_force_comparison.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_brute_force_comparison"
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_brute_force_comparison.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_brute_force_comparison.dir/build: tests/test_brute_force_comparison
.PHONY : tests/CMakeFiles/test_brute_force_comparison.dir/build

tests/CMakeFiles/test_brute_force_comparison.dir/clean:
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_brute_force_comparison.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_brute_force_comparison.dir/clean

tests/CMakeFiles/test_brute_force_comparison.dir/depend:
	cd /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/tests /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/tests /home/anish/Anish/Projects/GPU-Accelerated-Semantic-Similarity-Search/build/tests/CMakeFiles/test_brute_force_comparison.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/test_brute_force_comparison.dir/depend

