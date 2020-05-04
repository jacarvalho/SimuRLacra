#include "OMForceTorque.h"

#include <Rcs_typedef.h>
#include <Rcs_math.h>
#include <Rcs_MatNd.h>

#include <limits>
#include <vector>
#include <stdexcept>


namespace Rcs
{
  
  OMForceTorque::OMForceTorque(RcsGraph* graph, const char* sensorName, double max_force): max_force(max_force)
  {
      sensor = RcsGraph_getSensorByName(graph, sensorName);
      if (!sensor)
      {
          throw std::invalid_argument("Sensor not found: " + std::string(sensorName));
      }
      max_torque = std::numeric_limits<double>::infinity(); // [Nm]
  }
  
  OMForceTorque::~OMForceTorque() = default;
  
  unsigned int OMForceTorque::getStateDim() const
  {
      return 6;  // 3 forces, 3 torques
  }
  
  unsigned int OMForceTorque::getVelocityDim() const
  {
      // no derivative
      return 0;
  }
  
  void OMForceTorque::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
  {
      VecNd_copy(state, sensor->rawData->ele, getStateDim());
  }
  
  void OMForceTorque::getLimits(double* minState, double* maxState, double* maxVelocity) const
  {
      // Forces
      for (size_t i = 0; i < getStateDim()/2; ++i)
      {
          minState[i] = -max_force;
          maxState[i] = max_force;
      }
      // Torques
      for (size_t i = getStateDim()/2; i < getStateDim(); ++i)
      {
          minState[i] = -max_torque;
          maxState[i] = max_torque;
      }
  }
  
  std::vector<std::string> OMForceTorque::getStateNames() const
  {
      std::string sn = sensor->name;
      return {sn + "_Fx", sn + "_Fy", sn + "_Fz", sn + "_Tx", sn + "_Ty", sn + "_Tz",};
  }
  
} /* namespace Rcs */
