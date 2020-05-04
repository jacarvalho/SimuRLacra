#ifndef _OMCOLLISIONCOST_H_
#define _OMCOLLISIONCOST_H_

#include "OMComputedVelocity.h"
#include "../config/PropertySource.h"

namespace Rcs
{

class OMCollisionCost : public ObservationModel
//class OMCollisionCost : public OMComputedVelocity
{
public:

    explicit OMCollisionCost(RcsCollisionMdl* collisionMdl, double maxCollCost = 1e3);

    virtual ~OMCollisionCost();

    virtual void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const;
//    virtual void computeState(double* state, const MatNd *currentAction, double dt) const;

    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;

    virtual unsigned int getStateDim() const;

    virtual unsigned int getVelocityDim() const;

    std::vector<std::string> getStateNames() const override;

private:
    //! Rcs collision model (not owned!)
    RcsCollisionMdl* collisionMdl;

    double maxCollCost;
};

} /* namespace Rcs */

#endif //_OMCOLLISIONCOST_H_
