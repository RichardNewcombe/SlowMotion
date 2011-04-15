##############################################################################
# CUDA configuration

PROJECT("SlowMotion")


CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release )
ENDIF()

# Platform configuration vars
INCLUDE(SetPlatformVars)



#SET(BUILD_SHARED_LIBS TRUE)
#SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
#SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)


#######################################################
## Find required dependencies
FIND_PACKAGE(FREEGLUT REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(Threads REQUIRED)
#FIND_PACKAGE(freenect REQUIRED)
#FIND_PACKAGE(freenectsync REQUIRED)
FIND_PACKAGE(Boost 1.40 COMPONENTS thread filesystem REQUIRED)
FIND_PACKAGE(Pangolin REQUIRED)
FIND_PACKAGE(CVD REQUIRED)
FIND_PACKAGE(Cg REQUIRED)
FIND_PACKAGE(FlowLib REQUIRED)
#######################################################
## Sources

SET(
  SOURCES
  ./src/kernels/stereo.h
  ./src/kernels.cu
 ./src/utils.h
./src/utilsShared.h

)


CUDA_INCLUDE_DIRECTORIES(
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
#	${CMAKE_CURRENT_BINARY_DIR} # for configuration files
  )

# C++ only includes
INCLUDE_DIRECTORIES(
	#${CUDA_SDK_ROOT_DIR}/C/common/inc
	${CUDA_CUT_INCLUDE_DIR}
	${CUDA_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_CURRENT_BINARY_DIR} # for configuration files
	${FLOW_INCLUDE_DIRS}
	)


INCLUDE_DIRECTORIES(
  ${CUDA_SDK_ROOT_DIR}/common/inc
  ${CG_INCLUDE_PATH}
  ${Boost_INCLUDE_DIRS}
#  ${freenect_INCLUDE_DIR}
#  ${freenectsync_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIR}
  ${CVD_INCLUDE_DIR}
  ${USB_INCLUDE_DIRS}
)

LINK_LIBRARIES(
  ${ALL_LIBRARIES}
  ${LIBRARY_NAME}
  ${GLEW_LIBRARY}
  ${CG_LIBRARY}
  ${CG_GL_LIBRARY}
#  ${freenect_LIBRARY}
#  ${freenectsync_LIBRARY}
  ${Boost_LIBRARIES}
  ${Pangolin_LIBRARY}
  ${CVD_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT}
  ${FREEGLUT_LIBRARY}
  #libcudpp_x86_64.a
libcutil_x86_64.a
    -lGVars3
 ${FLOW_LIBRARIES}

)

LINK_DIRECTORIES(
  ${CUDA_SDK_ROOT_DIR}/lib
  ${CUDA_SDK_ROOT_DIR}/common/lib
  ${Boost_LIBRARY_DIRS}
  ${FLOW_LIBRARY_DIR}
)

# We need shared memory atomics "--maxrregcount=32""--ptxas-options=--opt-level=100"
SET(CUDA_NVCC_FLAGS  "-arch=sm_20" "--use_fast_math" "-O3"  "--ptxas-options=--verbose" "-keep"  )
ADD_DEFINITIONS(-DCUDA_BUILD -DBT_USE_DOUBLE_PRECISION -DCUDA_VERSION_32)

#SET(CMAKE_CXX_FLAGS "-Wall")


CUDA_ADD_EXECUTABLE(Stereo
   ${SOURCES}
  ./src/main.cpp
)







