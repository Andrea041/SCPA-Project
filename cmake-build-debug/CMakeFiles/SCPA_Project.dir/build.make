# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

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
CMAKE_COMMAND = /snap/clion/310/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /snap/clion/310/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pierfrancesco/Desktop/SCPA-Project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/SCPA_Project.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/SCPA_Project.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/SCPA_Project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SCPA_Project.dir/flags.make

CMakeFiles/SCPA_Project.dir/src/main_app.c.o: CMakeFiles/SCPA_Project.dir/flags.make
CMakeFiles/SCPA_Project.dir/src/main_app.c.o: /home/pierfrancesco/Desktop/SCPA-Project/src/main_app.c
CMakeFiles/SCPA_Project.dir/src/main_app.c.o: CMakeFiles/SCPA_Project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/SCPA_Project.dir/src/main_app.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/SCPA_Project.dir/src/main_app.c.o -MF CMakeFiles/SCPA_Project.dir/src/main_app.c.o.d -o CMakeFiles/SCPA_Project.dir/src/main_app.c.o -c /home/pierfrancesco/Desktop/SCPA-Project/src/main_app.c

CMakeFiles/SCPA_Project.dir/src/main_app.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/SCPA_Project.dir/src/main_app.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pierfrancesco/Desktop/SCPA-Project/src/main_app.c > CMakeFiles/SCPA_Project.dir/src/main_app.c.i

CMakeFiles/SCPA_Project.dir/src/main_app.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/SCPA_Project.dir/src/main_app.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pierfrancesco/Desktop/SCPA-Project/src/main_app.c -o CMakeFiles/SCPA_Project.dir/src/main_app.c.s

CMakeFiles/SCPA_Project.dir/src/functionsIO.c.o: CMakeFiles/SCPA_Project.dir/flags.make
CMakeFiles/SCPA_Project.dir/src/functionsIO.c.o: /home/pierfrancesco/Desktop/SCPA-Project/src/functionsIO.c
CMakeFiles/SCPA_Project.dir/src/functionsIO.c.o: CMakeFiles/SCPA_Project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/SCPA_Project.dir/src/functionsIO.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/SCPA_Project.dir/src/functionsIO.c.o -MF CMakeFiles/SCPA_Project.dir/src/functionsIO.c.o.d -o CMakeFiles/SCPA_Project.dir/src/functionsIO.c.o -c /home/pierfrancesco/Desktop/SCPA-Project/src/functionsIO.c

CMakeFiles/SCPA_Project.dir/src/functionsIO.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/SCPA_Project.dir/src/functionsIO.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pierfrancesco/Desktop/SCPA-Project/src/functionsIO.c > CMakeFiles/SCPA_Project.dir/src/functionsIO.c.i

CMakeFiles/SCPA_Project.dir/src/functionsIO.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/SCPA_Project.dir/src/functionsIO.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pierfrancesco/Desktop/SCPA-Project/src/functionsIO.c -o CMakeFiles/SCPA_Project.dir/src/functionsIO.c.s

CMakeFiles/SCPA_Project.dir/src/csrTool.c.o: CMakeFiles/SCPA_Project.dir/flags.make
CMakeFiles/SCPA_Project.dir/src/csrTool.c.o: /home/pierfrancesco/Desktop/SCPA-Project/src/csrTool.c
CMakeFiles/SCPA_Project.dir/src/csrTool.c.o: CMakeFiles/SCPA_Project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/SCPA_Project.dir/src/csrTool.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/SCPA_Project.dir/src/csrTool.c.o -MF CMakeFiles/SCPA_Project.dir/src/csrTool.c.o.d -o CMakeFiles/SCPA_Project.dir/src/csrTool.c.o -c /home/pierfrancesco/Desktop/SCPA-Project/src/csrTool.c

CMakeFiles/SCPA_Project.dir/src/csrTool.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/SCPA_Project.dir/src/csrTool.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pierfrancesco/Desktop/SCPA-Project/src/csrTool.c > CMakeFiles/SCPA_Project.dir/src/csrTool.c.i

CMakeFiles/SCPA_Project.dir/src/csrTool.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/SCPA_Project.dir/src/csrTool.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pierfrancesco/Desktop/SCPA-Project/src/csrTool.c -o CMakeFiles/SCPA_Project.dir/src/csrTool.c.s

CMakeFiles/SCPA_Project.dir/src/csrOperations.c.o: CMakeFiles/SCPA_Project.dir/flags.make
CMakeFiles/SCPA_Project.dir/src/csrOperations.c.o: /home/pierfrancesco/Desktop/SCPA-Project/src/csrOperations.c
CMakeFiles/SCPA_Project.dir/src/csrOperations.c.o: CMakeFiles/SCPA_Project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/SCPA_Project.dir/src/csrOperations.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/SCPA_Project.dir/src/csrOperations.c.o -MF CMakeFiles/SCPA_Project.dir/src/csrOperations.c.o.d -o CMakeFiles/SCPA_Project.dir/src/csrOperations.c.o -c /home/pierfrancesco/Desktop/SCPA-Project/src/csrOperations.c

CMakeFiles/SCPA_Project.dir/src/csrOperations.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/SCPA_Project.dir/src/csrOperations.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pierfrancesco/Desktop/SCPA-Project/src/csrOperations.c > CMakeFiles/SCPA_Project.dir/src/csrOperations.c.i

CMakeFiles/SCPA_Project.dir/src/csrOperations.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/SCPA_Project.dir/src/csrOperations.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pierfrancesco/Desktop/SCPA-Project/src/csrOperations.c -o CMakeFiles/SCPA_Project.dir/src/csrOperations.c.s

CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.o: CMakeFiles/SCPA_Project.dir/flags.make
CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.o: /home/pierfrancesco/Desktop/SCPA-Project/src/hll_ellpack_Tool.c
CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.o: CMakeFiles/SCPA_Project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.o -MF CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.o.d -o CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.o -c /home/pierfrancesco/Desktop/SCPA-Project/src/hll_ellpack_Tool.c

CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pierfrancesco/Desktop/SCPA-Project/src/hll_ellpack_Tool.c > CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.i

CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pierfrancesco/Desktop/SCPA-Project/src/hll_ellpack_Tool.c -o CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.s

CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.o: CMakeFiles/SCPA_Project.dir/flags.make
CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.o: /home/pierfrancesco/Desktop/SCPA-Project/src/hll_Operations.c
CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.o: CMakeFiles/SCPA_Project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.o -MF CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.o.d -o CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.o -c /home/pierfrancesco/Desktop/SCPA-Project/src/hll_Operations.c

CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pierfrancesco/Desktop/SCPA-Project/src/hll_Operations.c > CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.i

CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pierfrancesco/Desktop/SCPA-Project/src/hll_Operations.c -o CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.s

# Object files for target SCPA_Project
SCPA_Project_OBJECTS = \
"CMakeFiles/SCPA_Project.dir/src/main_app.c.o" \
"CMakeFiles/SCPA_Project.dir/src/functionsIO.c.o" \
"CMakeFiles/SCPA_Project.dir/src/csrTool.c.o" \
"CMakeFiles/SCPA_Project.dir/src/csrOperations.c.o" \
"CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.o" \
"CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.o"

# External object files for target SCPA_Project
SCPA_Project_EXTERNAL_OBJECTS =

SCPA_Project: CMakeFiles/SCPA_Project.dir/src/main_app.c.o
SCPA_Project: CMakeFiles/SCPA_Project.dir/src/functionsIO.c.o
SCPA_Project: CMakeFiles/SCPA_Project.dir/src/csrTool.c.o
SCPA_Project: CMakeFiles/SCPA_Project.dir/src/csrOperations.c.o
SCPA_Project: CMakeFiles/SCPA_Project.dir/src/hll_ellpack_Tool.c.o
SCPA_Project: CMakeFiles/SCPA_Project.dir/src/hll_Operations.c.o
SCPA_Project: CMakeFiles/SCPA_Project.dir/build.make
SCPA_Project: /usr/lib/x86_64-linux-gnu/libcjson.so.1.7.18
SCPA_Project: /usr/lib/gcc/x86_64-linux-gnu/12/libgomp.so
SCPA_Project: /usr/lib/x86_64-linux-gnu/libpthread.a
SCPA_Project: CMakeFiles/SCPA_Project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking C executable SCPA_Project"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SCPA_Project.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SCPA_Project.dir/build: SCPA_Project
.PHONY : CMakeFiles/SCPA_Project.dir/build

CMakeFiles/SCPA_Project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SCPA_Project.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SCPA_Project.dir/clean

CMakeFiles/SCPA_Project.dir/depend:
	cd /home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pierfrancesco/Desktop/SCPA-Project /home/pierfrancesco/Desktop/SCPA-Project /home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug /home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug /home/pierfrancesco/Desktop/SCPA-Project/cmake-build-debug/CMakeFiles/SCPA_Project.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/SCPA_Project.dir/depend

