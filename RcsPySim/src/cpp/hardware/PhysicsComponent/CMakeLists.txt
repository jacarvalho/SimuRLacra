IF(NOT TARGET RcsPhysicsComponent)

SET(SRC
PhysicsSimulationComponent.cpp
)

ADD_LIBRARY(RcsPhysicsComponent ${LIBRARY_LINKING_TYPE} ${SRC})
TARGET_LINK_LIBRARIES(RcsPhysicsComponent RcsPhysics)
TARGET_INCLUDE_DIRECTORIES(RcsPhysicsComponent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

ENDIF(NOT TARGET RcsPhysicsComponent)
