#ifndef _OMCOMPUTEDVELOCITY_H
#define _OMCOMPUTEDVELOCITY_H

#include "ObservationModel.h"

namespace Rcs
{

/*!
 * An observation model that computes the velocity using finite differences.
 * Use this observation model if the velocity can not be or is not observed directly from the simulation.
 */
class OMComputedVelocity : public ObservationModel
{
public:
    OMComputedVelocity();

    virtual ~OMComputedVelocity();

    // DO NOT OVERRIDE!
    void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const final;

    /**
     * Overridden to initialize lastState.
     * If a subclass wants to override this, make sure to call the base method. Since this method has to call
     * computeState, make sure any relevant initialization is done before.
     */
    virtual void reset();

    /**
     * Implement to fill the observation vector with the observed state values. The velocity will be computed automatically.
     * @param[out] state state observation vector to fill, has getStateDim() elements.
     * @param[in] currentAction action in current step. May be NULL if not available.
     * @param[in] dt time step since the last observation has been taken. Will be 0 if called during reset().
     */
    virtual void computeState(double* state, const MatNd *currentAction, double dt) const = 0;

private:
    // state during last call to computeObservation
    MatNd* lastState;
};

} /* namespace Rcs */

#endif //_OMCOMPUTEDVELOCITY_H
