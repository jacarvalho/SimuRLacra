#ifndef _OMTASKSPACEDISCREPANCY_H
#define _OMTASKSPACEDISCREPANCY_H

#include "ObservationModel.h"


namespace Rcs
{

class ControllerBase;

/**
 * ObservationModel computing the discrepancy between a body's position in the desired graph (owned by the controller)
 * and the current graph (owned by the config) in task space.
 */
class OMTaskSpaceDiscrepancy : public ObservationModel
{
public:
    /**
     * Create from action model.
     * NOTE: assumes that the task activation action model wraps a IK-based action model.
     */
    explicit OMTaskSpaceDiscrepancy(
        const char* bodyName,
        const RcsGraph* controllerGraph,
        const RcsGraph* configGraph,
        double maxDiscrepancy = 1.
    );

    virtual ~OMTaskSpaceDiscrepancy();

    // not copy- or movable - klocwork doesn't pick up the inherited ones. RCSPYSIM_NOCOPY_NOMOVE(OMDynamicalSystemDiscrepancy)

    virtual unsigned int getStateDim() const;

    unsigned int getVelocityDim() const override;

    void getLimits(double* minState, double* maxState, double* maxVelocity) const override;

    virtual void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const;

    void reset() override;

    virtual std::vector<std::string> getStateNames() const;

private:
    // Body of interest to determine the discrepancy between, e.g. the end-effector
    RcsBody* bodyController;
    RcsBody* bodyConfig;

    // We make the design decision that more than maxDiscrepancy difference are not acceptable in any case
    double maxDiscrepancy;  // default is 1 [m, m/s, rad, rad/s]
};

} /* namespace Rcs */

#endif //_OMTASKSPACEDISCREPANCY_H
