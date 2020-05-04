#include "OMTaskSpaceDiscrepancy.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <Rcs_Vec3d.h>
#include <Rcs_VecNd.h>

#include <limits>


namespace Rcs
{
  
  OMTaskSpaceDiscrepancy::OMTaskSpaceDiscrepancy(
      const char* bodyName,
      const RcsGraph* controllerGraph,
      const RcsGraph* configGraph,
      double maxDiscrepancy
      ) : maxDiscrepancy(maxDiscrepancy)
  {
      bodyController = RcsGraph_getBodyByName(controllerGraph, bodyName);
      RCHECK(bodyController);
      bodyConfig = RcsGraph_getBodyByName(configGraph, bodyName);
      RCHECK(bodyConfig);
  }
  
  OMTaskSpaceDiscrepancy::~OMTaskSpaceDiscrepancy()
  {
      // Pointer on bodies are destroyed by the graph
  }
  
  unsigned int OMTaskSpaceDiscrepancy::getStateDim() const
  {
      return 3; // only Cartesian position difference
  }
  
  unsigned int OMTaskSpaceDiscrepancy::getVelocityDim() const
  {
      return 0;  // does not have a velocity field
  }
  
  void OMTaskSpaceDiscrepancy::getLimits(double* minState, double* maxState, double* maxVelocity) const
  {
      unsigned int sd = getStateDim();
      VecNd_setElementsTo(minState, -maxDiscrepancy, sd);
      VecNd_setElementsTo(maxState, maxDiscrepancy, sd);
      VecNd_setElementsTo(maxVelocity, 0., getVelocityDim());
  }
  
  void OMTaskSpaceDiscrepancy::computeObservation(
      double* state,
      double* velocity,
      const MatNd* currentAction,
      double dt) const
  {
      // Get the difference (desired - current)
      Vec3d_sub(state, bodyController->A_BI->org, bodyConfig->A_BI->org);
    
      // Print if debug level is exceeded
      REXEC(7)
      {
          std::cout << "Task space discrepancy: " << state << std::endl;
      }
  }
  
  void OMTaskSpaceDiscrepancy::reset()
  {
//      RcsBody* bodyController = RcsGraph_getBodyByName(controllerGraph, bodyName);
//      RcsBody* bodyConfig = RcsGraph_getBodyByName(configGraph, bodyName);
  }
  
  std::vector<std::string> OMTaskSpaceDiscrepancy::getStateNames() const
  {
      return {"DiscrepTS_X", "DiscrepTS_Y", "DiscrepTS_Z"};
  }
  
} /* namespace Rcs */