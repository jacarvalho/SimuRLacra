#ifndef _OMGOALDISTANCE_H_
#define _OMGOALDISTANCE_H_

#include "ObservationModel.h"
#include "../action/AMTaskActivation.h"


namespace Rcs
{

class ControllerBase;

/**
 * ObservationModel wrapping multiple AMTaskActivation to compute the distances to the individuals goals but not
 * the rate of change of these goal distances.
 */
class OMGoalDistance : public ObservationModel
{
public:
    /**
     * Create from action model.
     * NOTE: assumes that the task activation action model wraps a IK-based action model.
     */
    OMGoalDistance(AMTaskActivation* actionModel);

    virtual ~OMGoalDistance();

    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(OMGoalDistance)

    virtual unsigned int getStateDim() const;

    virtual unsigned int getVelocityDim() const;

    virtual void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const;

    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;

    virtual std::vector<std::string> getStateNames() const;


private:
    // Task activation action model, provides the tasks (not owned)
    AMTaskActivation* actionModel;

    // Controller to extract the task space state
    ControllerBase* controller;

    // Limits
    double maxDistance; // by default infinity
};

} /* namespace Rcs */

#endif /* _OMGOALDISTANCE_H_ */
