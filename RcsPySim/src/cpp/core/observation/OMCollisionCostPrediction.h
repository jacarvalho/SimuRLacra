#ifndef _OMCOLLISIONCOSTPREDICTION_H_
#define _OMCOLLISIONCOSTPREDICTION_H_

#include "ObservationModel.h"
#include "../config/PropertySource.h"
#include "../action/ActionModel.h"

namespace Rcs
{

class OMCollisionCostPrediction : public ObservationModel
{
public:

    OMCollisionCostPrediction(RcsGraph *graph, RcsCollisionMdl* collisionMdl, const ActionModel *actionModel,
                              size_t horizon = 10, double maxCollCost = 1e4);

    virtual ~OMCollisionCostPrediction();

    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(OMCollisionCostPrediction)

    virtual void computeObservation(double* state, double *velocity, const MatNd *currentAction, double dt) const;

    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;

    virtual unsigned int getStateDim() const;

    virtual unsigned int getVelocityDim() const;

    std::vector<std::string> getStateNames() const override;

private:
    //! Graph to observe (not owned)
    RcsGraph* realGraph;

    //! Graph copy for prediction
    RcsGraph* predictGraph;

    //! Action model copy for prediction
    ActionModel* predictActionModel;

    //! Rcs collision model
    RcsCollisionMdl* collisionMdl;

    //! Time horizon for the predicted collision costs (horizon = 1 mean the current step plus one step ahead)
    size_t horizon;
  
  double maxCollCost;
};

}

#endif //_OMCOLLISIONCOSTPREDICTION_H_
