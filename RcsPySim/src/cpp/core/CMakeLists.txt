# Include source files
set(CORE_SRCS
    
    RcsPyBot.cpp
    RcsSimEnv.cpp
    
    ExperimentConfig.cpp
    ECBallOnPlate.cpp
    ECBoxFlipping.cpp
    ECBoxLifting.cpp
    ECBoxLiftingSimple.cpp
    ECBoxShelving.cpp
    ECMPBlending.cpp
    ECPlanar3Link.cpp
    ECPlanarInsert.cpp
    ECQuanserQube.cpp
    ECTargetTracking.cpp

    DataLogger.cpp

    config/PropertySource.cpp
    config/PropertySourceXml.cpp

    control/MLPPolicy.cpp
    control/ControlPolicy.cpp

    initState/InitStateSetter.cpp
    initState/ISSBallOnPlate.cpp
    initState/ISSBoxFlipping.cpp
    initState/ISSBoxLifting.cpp
    initState/ISSBoxLiftingSimple.cpp
    initState/ISSBoxShelving.cpp
    initState/ISSMPBlending.cpp
    initState/ISSPlanar3Link.cpp
    initState/ISSPlanarInsert.cpp
    initState/ISSQuanserQube.cpp

    action/ActionModel.cpp
    action/ActionModelIK.cpp
    action/AMIntegrate1stOrder.cpp
    action/AMIntegrate2ndOrder.cpp
    action/AMJointControl.cpp
    action/AMJointControlPosition.cpp
    action/AMJointControlAcceleration.cpp
    action/AMNormalized.cpp
    action/AMPlateAngPos.cpp
    action/AMPlatePos5D.cpp
    action/AMTaskActivation.cpp
    action/DynamicalSystem.cpp

    observation/ObservationModel.cpp
    observation/OMCombined.cpp
    observation/OMNormalized.cpp
    observation/OMComputedVelocity.cpp
    observation/OMTask.cpp
    observation/OMTaskPositions.cpp
    observation/OMPartial.cpp
    observation/OMBodyStateAngular.cpp
    observation/OMBodyStateLinear.cpp
    observation/OMJointState.cpp
    observation/OMBallPos.cpp
    observation/OMGoalDistance.cpp
    observation/OMForceTorque.cpp
    observation/OMCollisionCost.cpp
    observation/OMCollisionCostPrediction.cpp
    observation/OMDynamicalSystemDiscrepancy.cpp
    observation/OMTaskSpaceDiscrepancy.cpp
    observation/OMManipulabilityIndex.cpp

    physics/PhysicsParameterManager.cpp
    physics/BodyParamInfo.cpp
    physics/PhysicsParameterDescriptor.cpp
    physics/PPDBodyPosition.cpp
    physics/PPDBoxExtents.cpp
    physics/PPDCompound.cpp
    physics/PPDMassProperties.cpp
    physics/PPDMaterialProperties.cpp
    physics/PPDRodLength.cpp
    physics/PPDSphereRadius.cpp
    physics/ForceDisturber.cpp
    physics/vortex_log.cpp
    
    util/BoxSpace.cpp
    util/BoxSpaceProvider.cpp
    util/integrator.cpp
)

# Add pytorch specific sources if available
IF(TORCH_FOUND)
#IF(USE_LIBTORCH)
  LIST(APPEND CORE_SRCS
    control/TorchPolicy.cpp
  )
#ENDIF()
ENDIF()

add_library(RcsEnvCore SHARED ${CORE_SRCS})

# C++11
target_compile_options(RcsEnvCore PUBLIC ${PYBIND11_CPP_STANDARD})

# Include dir to interface so that dependent users can find the files.
target_include_directories(RcsEnvCore INTERFACE ${CMAKE_CURRENT_LIST_DIR})

# Set rcs config dir
target_compile_definitions(RcsEnvCore PUBLIC "-DRCS_CONFIG_DIR=\"${RCS_CONFIG_DIR}\"")

# Depend on imported Rcs Targets
target_link_libraries(RcsEnvCore PUBLIC RcsCore -Wl,--no-as-needed RcsPhysics -Wl,--as-needed)
# Depend on Eigen
target_include_directories(RcsEnvCore PUBLIC SYSTEM ${EIGEN3_INCLUDE_DIR})

# Check whether RcsGraphics is available
IF (${Rcs_RcsGraphics_FOUND})
  # Link to it and define graphics enabler
  target_link_libraries(RcsEnvCore PUBLIC RcsGraphics)
  target_compile_definitions(RcsEnvCore PUBLIC "-DGRAPHICS_AVAILABLE")
ENDIF (${Rcs_RcsGraphics_FOUND})

# Check whether RcsGui is available
IF (${Rcs_RcsGui_FOUND})
  # Link to it and define graphics enabler
  target_link_libraries(RcsEnvCore PUBLIC RcsGui)
  target_link_libraries(RcsEnvCore PRIVATE ${QWT_LIBRARIES})
  target_include_directories(RcsEnvCore PRIVATE ${QWT_INCLUDE_DIRS})
  target_compile_definitions(RcsEnvCore PUBLIC "-DGUI_AVAILABLE")
  set_target_properties(RcsEnvCore PROPERTIES AUTOMOC TRUE)
ENDIF (${Rcs_RcsGui_FOUND})

# Depend on pytorch if available.
IF(TORCH_FOUND)
  # Use imported torch library, automatically sets up include dirs & compile options
  TARGET_LINK_LIBRARIES(RcsEnvCore PUBLIC torch)
ENDIF()

# Depend on physics sim headers if available.
# We only need this for the vortex log hack.
IF (USE_VORTEX)
  # Build the vortex sub-library
  FIND_PACKAGE(Vortex)

  TARGET_INCLUDE_DIRECTORIES(RcsEnvCore SYSTEM PRIVATE ${VORTEX_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(RcsEnvCore PRIVATE ${VORTEX_LIBRARIES})
  # Also add a definition for using vortex at all, in order to exclude vortex-specific code if not set
  TARGET_COMPILE_DEFINITIONS(RcsEnvCore PRIVATE ${VORTEX_DEFINITIONS} "-DUSE_VORTEX")
ENDIF (USE_VORTEX)
