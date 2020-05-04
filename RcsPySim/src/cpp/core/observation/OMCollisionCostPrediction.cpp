#include "OMCollisionCostPrediction.h"
#include "OMCollisionCost.h"

#include <Rcs_collisionModel.h>
#include <Rcs_typedef.h>
#include <Rcs_VecNd.h>
#include <Rcs_macros.h>

#include <stdexcept>

namespace Rcs
{
  
  OMCollisionCostPrediction::OMCollisionCostPrediction(RcsGraph* graph, RcsCollisionMdl* collisionMdl,
                                                       const ActionModel* actionModel, size_t horizon,
                                                       double maxCollCost) : realGraph(graph), horizon(horizon),
                                                                             maxCollCost(maxCollCost)
  {
      // Copy graph and action model for prediction
      predictGraph = RcsGraph_clone(graph);
      predictActionModel = actionModel->clone(predictGraph);
      
      // Copy collision model for predict graph.
      this->collisionMdl = RcsCollisionModel_clone(collisionMdl, predictGraph);
  }
  
  
  OMCollisionCostPrediction::~OMCollisionCostPrediction()
  {
      // Destroy cloned collision model
      RcsCollisionModel_destroy(collisionMdl);
      
      delete predictActionModel;
      RcsGraph_destroy(predictGraph);
  }
  
  unsigned int OMCollisionCostPrediction::getStateDim() const
  {
      return 1;
  }
  
  unsigned int OMCollisionCostPrediction::getVelocityDim() const
  {
      return 0;
  }
  
  void OMCollisionCostPrediction::computeObservation(double* state, double* velocity, const MatNd* currentAction,
                                                     double dt) const
  {
      // NOTE: prediction doesn't update q_dot, since we don't simulate it and it's not relevant for the collision model.
      
      // Reset prediction state
      RcsGraph_setState(predictGraph, realGraph->q, realGraph->q_dot);
      predictActionModel->reset();
      
      // Compute cost for initial step
      RcsCollisionModel_compute(collisionMdl);
      double predCostSum = RcsCollisionMdl_cost(collisionMdl);
      
      // This OM can only predict the future collision costs if the current action was given
      if (currentAction != NULL)
      {
          // Execute prediction
          for (size_t i = 0; i <= horizon; ++i)
          {
              // Compute action from action model
              // Passing NULL to q_dot_des and T_des is not strictly allowed, but since we can't support them, this
              // will at least cause an error with unsupported action models.
              predictActionModel->computeCommand(predictGraph->q, predictGraph->q_dot, NULL, currentAction, dt);
              
              // Update graph using forward kinematics (no physics simulation)
              RcsGraph_setState(predictGraph, NULL, predictGraph->q_dot);
              
              // Sum up step cost
              RcsCollisionModel_compute(collisionMdl);
              predCostSum += RcsCollisionMdl_cost(collisionMdl);
          }
          // Compute average
          predCostSum /= (horizon + 1.);
      }
      // The state is the average collision cost
      state[0] = predCostSum;
  }
  
  std::vector<std::string> OMCollisionCostPrediction::getStateNames() const
  {
      return {"PredCollCost_h" + std::to_string(horizon)};
  }
  
  void OMCollisionCostPrediction::getLimits(double* minState, double* maxState, double* maxVelocity) const
  {
//    ObservationModel::getLimits(minState, maxState, maxVelocity);
      VecNd_setZero(minState, getStateDim()); // minimum cost is 0
      VecNd_setElementsTo(maxState, maxCollCost, getStateDim());  // maximum cost (theoretically infinite)
  }
    
} /* namespace Rcs */
