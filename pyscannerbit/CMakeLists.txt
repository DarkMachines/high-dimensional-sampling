# GAMBIT: Global and Modular BSM Inference Tool
#************************************************
# \file
#
#  CMake configuration script for building
#  the python interface to ScannerBit
#
#************************************************
#
#  Authors (add name and date if you modify):
#
#  \author Ben Farmer
#          (ben.farmer@gmail.com)
#  \date 2017 Nov, 2019 Nov
#
#************************************************

# Require a minimum cmake version of 2.8.5
cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
project(pyscannerbit)

# Make sure -fPIC added when building subprojects like yaml-cpp
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build GAMBIT
set(GAMBIT_DIR "${PROJECT_SOURCE_DIR}/pyscannerbit/scannerbit/untar/ScannerBit")
set(GAMBIT_RUN_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}") # Final package installation path
#set(LIB_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pyscannerbit") # Temporary build path (unused?)

message("Setting GAMBIT_RUN_DIR to: ${GAMBIT_RUN_DIR}")
add_subdirectory("${GAMBIT_DIR}")

#--- Post-build copying
# Ok the build seems to be working, but now I need to copy all the scanner stuff
# into the final python package install directory.
# There might be a "cleaner" CMake way to do this... I will do it this "dumb"
# way for now.

add_custom_target(post_build DEPENDS scan_python)

# Copies stuff after scan_python target is built
set(copy_target post_build)
function(copy_file in out)
   message("Generating post-build instructions to copy ${in} to ${out}")
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying ${in} to ${out}...")
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${out})
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${in} ${out})
endfunction(copy_file)

function(copy_dir in out)
   message("Generating post-build instructions to copy ${in} to ${out}")
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying ${in} to ${out}...")
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${out})
   add_custom_command(
        TARGET ${copy_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${in} ${out})
endfunction(copy_dir)

# Copy the ScannerBit 'python' directory that will contain the main 'ScannerBit.so' python-usable library
copy_dir(${GAMBIT_DIR}/ScannerBit/python
         ${GAMBIT_RUN_DIR}/ScannerBit/python)

# Copy the scanner plugins directory
copy_dir(${GAMBIT_DIR}/ScannerBit/lib
         ${GAMBIT_RUN_DIR}/ScannerBit/lib)

# Copy the GAMBIT config directory (contains some config files that scannerbit reads)
copy_dir(${GAMBIT_DIR}/config
         ${GAMBIT_RUN_DIR}/config)

# Copy the GAMBIT scratch directory (contains some more config files that scannerbit reads)
copy_dir(${GAMBIT_DIR}/scratch
         ${GAMBIT_RUN_DIR}/scratch)


# Copy each of the scanner 'backend' shared libraries
set(MULTINEST_VERSION 3.11)
set(POLYCHORD_VERSION 1.15)
set(DIVER_VERSION 1.0.4)

# Multinest
copy_file(${GAMBIT_DIR}/ScannerBit/installed/multinest/${MULTINEST_VERSION}/libnest3.so
          ${GAMBIT_RUN_DIR}/ScannerBit/installed/multinest/${MULTINEST_VERSION})

# Polychord
copy_file(${GAMBIT_DIR}/ScannerBit/installed/polychord/${POLYCHORD_VERSION}/lib/libchord.so
          ${GAMBIT_RUN_DIR}/ScannerBit/installed/polychord/${POLYCHORD_VERSION}/lib)

# Diver
copy_file(${GAMBIT_DIR}/ScannerBit/installed/diver/${DIVER_VERSION}/lib/libdiver.so
          ${GAMBIT_RUN_DIR}/ScannerBit/installed/diver/${DIVER_VERSION}/lib)

# Leaving GreAT out on purpose since it needs ROOT and that is just too painful to deal with.

# Add a couple of __init__.py files to make the module work properly


