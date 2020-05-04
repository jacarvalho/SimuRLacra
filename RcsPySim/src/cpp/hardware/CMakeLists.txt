# The components are not part of the public Rcs yet, so use copied versions:
add_subdirectory(ViewerComponent)
add_subdirectory(PhysicsComponent)


add_executable(TestBotSim TestBotSim.cpp)
target_link_libraries(TestBotSim PRIVATE RcsEnvCore RcsViewerComponent RcsPhysicsComponent z)

IF(TARGET RcsROSComponents)

  add_executable(TestBallTracking TestBallTracking.cpp)
  target_link_libraries(TestBallTracking PRIVATE RcsEnvCore RcsViewerComponent RcsROSComponents)

ENDIF()

IF(TARGET RcsErgo)
  add_executable(ECSRobo ECSRobo.cpp PolicyComponent.cpp)
  target_link_libraries(ECSRobo PRIVATE RcsEnvCore RcsErgo RcsEventGui RcsComponents)
  IF (UNIX)
    TARGET_COMPILE_DEFINITIONS(ECSRobo PRIVATE -DECS_WITH_ROBO)
  ENDIF(UNIX)
ENDIF()
