# Find Hikvision MVS SDK (x86_64)
# Exports:
#   MVS_FOUND
#   MVS_INCLUDE_DIRS
#   MVS_LIBS

set(MVS_ROOT_PATHS
    /opt/MVS
    /usr/local/MVS
    /usr/MVS
)

find_path(MVS_INCLUDE_DIRS
    NAMES MvCameraControl.h
    PATH_SUFFIXES include
    PATHS ${MVS_ROOT_PATHS}
)

find_library(MVS_LIB_CAMERA_CONTROL
    NAMES MvCameraControl
    PATH_SUFFIXES lib lib/64
    PATHS ${MVS_ROOT_PATHS}
)

find_library(MVS_LIB_FORMAT_CONVERT
    NAMES FormatConversion MvImageConvert libFormatConversion
    PATH_SUFFIXES lib lib/64
    PATHS ${MVS_ROOT_PATHS}
)

find_library(MVS_LIB_GIGE
    NAMES MVGigEVisionSDK MvGigE libMVGigEVisionSDK
    PATH_SUFFIXES lib lib/64
    PATHS ${MVS_ROOT_PATHS}
)

find_library(MVS_LIB_USB
    NAMES MvUsb3vTL MvUsb3
    PATH_SUFFIXES lib lib/64
    PATHS ${MVS_ROOT_PATHS}
)

set(MVS_LIBS
    ${MVS_LIB_CAMERA_CONTROL}
    ${MVS_LIB_FORMAT_CONVERT}
    ${MVS_LIB_GIGE}
    ${MVS_LIB_USB}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MVS
    REQUIRED_VARS MVS_INCLUDE_DIRS MVS_LIB_CAMERA_CONTROL
)

mark_as_advanced(
    MVS_INCLUDE_DIRS
    MVS_LIB_CAMERA_CONTROL
    MVS_LIB_FORMAT_CONVERT
    MVS_LIB_GIGE
    MVS_LIB_USB
)

if(MVS_FOUND)
    message(STATUS "Found Hikvision MVS SDK at /opt/MVS")
    message(STATUS "  Include:    ${MVS_INCLUDE_DIRS}")
    message(STATUS "  Libraries:  ${MVS_LIBS}")
else()
    message(FATAL_ERROR "Could not find MVS SDK. Install it under /opt/MVS")
endif()
