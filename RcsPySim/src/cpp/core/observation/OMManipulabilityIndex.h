#ifndef RCSPYSIM_OMMANIPULABILITYINDEX_H
#define RCSPYSIM_OMMANIPULABILITYINDEX_H

#include "ObservationModel.h"
#include "../action/ActionModelIK.h"

namespace Rcs {

/**
 * Observes the manipulability index of the graph.
 */
class OMManipulabilityIndex : public ObservationModel
{
private:
    // controller used to compute the manipulability.
    ControllerBase* controller;

    // observed graph
    RcsGraph* observedGraph;

public:
    /**
     * Create from action model
     * @param ikModel action model to copy controller tasks from
     * @param observeCurrent true to observe the current graph instead of the desired graph.
     */
    explicit OMManipulabilityIndex(ActionModelIK* ikModel, bool observeCurrent=false);

    ~OMManipulabilityIndex() override;

    unsigned int getStateDim() const override;

    unsigned int getVelocityDim() const override;

    void computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const override;

    std::vector<std::string> getStateNames() const override;
};

}

#endif //RCSPYSIM_OMMANIPULABILITYINDEX_H
