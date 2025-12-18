# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS
   "/Users/sofakapanova/ppc1/ppc-2025-processes-mathematics/3rdparty/json")
  file(MAKE_DIRECTORY
       "/Users/sofakapanova/ppc1/ppc-2025-processes-mathematics/3rdparty/json")
endif()
file(
  MAKE_DIRECTORY
  "/Users/sofakapanova/ppc1/ppc-2025-processes-mathematics/ppc_json/build"
  "/Users/sofakapanova/ppc1/ppc-2025-processes-mathematics/ppc_json/install"
  "/Users/sofakapanova/ppc1/ppc-2025-processes-mathematics/ppc_json/tmp"
  "/Users/sofakapanova/ppc1/ppc-2025-processes-mathematics/ppc_json/src/ppc_json-stamp"
  "/Users/sofakapanova/ppc1/ppc-2025-processes-mathematics/ppc_json/src"
  "/Users/sofakapanova/ppc1/ppc-2025-processes-mathematics/ppc_json/src/ppc_json-stamp"
)

set(configSubDirs)
foreach(subDir IN LISTS configSubDirs)
  file(
    MAKE_DIRECTORY
    "/Users/sofakapanova/ppc1/ppc-2025-processes-mathematics/ppc_json/src/ppc_json-stamp/${subDir}"
  )
endforeach()
if(cfgdir)
  file(
    MAKE_DIRECTORY
    "/Users/sofakapanova/ppc1/ppc-2025-processes-mathematics/ppc_json/src/ppc_json-stamp${cfgdir}"
  ) # cfgdir has leading slash
endif()
