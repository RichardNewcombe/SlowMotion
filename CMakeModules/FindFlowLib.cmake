# - Try to find libflow
#
#  FLOW_FOUND - system has libflow
#  FLOW_INCLUDE_DIRS - the libflow include directories
#  FLOW_LIBRARIES - link these to use libflow
#  FLOW_LIBRARY_DIR - link these to use libflow


FIND_PATH(
  FLOW_INCLUDE_DIR
  NAMES fl/flowlib.h
  PATHS
    ${CMAKE_SOURCE_DIR}/../vmlibraries/include
    /usr/include
    /usr/local/include
)

FIND_PATH(
  IU_INCLUDE_DIR
  NAMES iu/iucore.h
  PATHS
    ${CMAKE_SOURCE_DIR}/../vmlibraries/include
    /usr/include
    /usr/local/include
)


FIND_PATH(
  FLOW_LIBRARY_DIR
  NAMES libiucore.so
  PATHS
    ${CMAKE_SOURCE_DIR}/../vmlibraries/lib
    /usr/lib
    /usr/local/lib
)




IF(FLOW_INCLUDE_DIR AND IU_INCLUDE_DIR AND FLOW_LIBRARY_DIR)
  SET(FLOW_INCLUDE_DIRS ${FLOW_INCLUDE_DIR};${IU_INCLUDE_DIR};${IU_INCLUDE_DIR}/iu;)
  SET(FLOW_LIBRARIES flow iucore iugui iuiopgm iuio )
  SET(FLOW_FOUND TRUE)
ENDIF()


IF(FLOW_FOUND)
   IF(NOT FLOW_FIND_QUIETLY)
      MESSAGE(STATUS "Found Flow: ${FLOW_LIBRARIES}")
   ENDIF()
ELSE()
   IF(FLOW_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find Flow ")
   ENDIF()
ENDIF()
