#ifndef _OMDYNAMICALSYSTEMDISCREPANCY_H_
#define _OMDYNAMICALSYSTEMDISCREPANCY_H_

#include "ObservationModel.h"
#include "../action/AMTaskActivation.h"


namespace Rcs
{

class ControllerBase;

/**
 * ObservationModel wrapping multiple AMTaskActivation to compute the discrepancies between the task space changes
 * commanded by the DS and the ones executed by the robot.
 */
  class OMDynamicalSystemDiscrepancy : public ObservationModel
{
public:
    /**
     * Create from action model.
     * NOTE: assumes that the task activation action model wraps a IK-based action model.
     */
    explicit OMDynamicalSystemDiscrepancy(AMTaskActivation* actionModel);
    
    virtual ~OMDynamicalSystemDiscrepancy();
    
    // not copy- or movable - klocwork doesn't pick up the inherited ones. RCSPYSIM_NOCOPY_NOMOVE(OMDynamicalSystemDiscrepancy)

    virtual unsigned int getStateDim() const;
  
    unsigned int getVelocityDim() const override;

    virtual void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const;

    void reset() override;

    virtual std::vector<std::string> getStateNames() const;


private:
    // Task activation action model, provides the tasks (not owned)
    AMTaskActivation* actionModel;

    // Controller to extract the task space state
    ControllerBase* controller;

    // last task space state
    MatNd* x_curr;
};

} /* namespace Rcs */

#endif /* _OMDYNAMICALSYSTEMDISCREPANCY_H_ */
