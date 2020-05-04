#include "OMComputedVelocity.h"

#include <Rcs_VecNd.h>

namespace Rcs
{
  
  OMComputedVelocity::OMComputedVelocity() : lastState(NULL)
  {}
  
  OMComputedVelocity::~OMComputedVelocity()
  {
      // destroy last state mat
      MatNd_destroy(lastState);
  }
  
  void
  OMComputedVelocity::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
  {
      // compute current state
      computeState(state, currentAction, dt);
      
      // compute velocity as forward derivative
      VecNd_sub(velocity, state, lastState->ele, getStateDim());
      VecNd_constMulSelf(velocity, 1/dt, getStateDim());
      
      // save state for next step
      VecNd_copy(lastState->ele, state, getStateDim());
  }
  
  void OMComputedVelocity::reset()
  {
      // create last state storage if needed
      if (lastState == NULL)
      {
          lastState = MatNd_create(getStateDim(), 1);
      }
      // initialize last state storage.
      computeState(lastState->ele, NULL, 0.0);
  }
  
} /* namespace Rcs */
