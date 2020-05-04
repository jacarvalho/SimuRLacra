#include <Rcs_typedef.h>
#include "OMManipulabilityIndex.h"

Rcs::OMManipulabilityIndex::OMManipulabilityIndex(Rcs::ActionModelIK* ikModel, bool observeCurrent)
{
    this->controller = new ControllerBase(*ikModel->getController());
    if (observeCurrent)
    {
        observedGraph = ikModel->getGraph();
    }
    else
    {
        observedGraph = ikModel->getDesiredGraph();
    }
}

Rcs::OMManipulabilityIndex::~OMManipulabilityIndex()
{
    delete controller;
}

unsigned int Rcs::OMManipulabilityIndex::getStateDim() const
{
    return 1;
}

unsigned int Rcs::OMManipulabilityIndex::getVelocityDim() const
{
    return 0;
}

void Rcs::OMManipulabilityIndex::computeObservation(double* state, double* velocity, const MatNd* currentAction,
                                                    double dt) const
{
    RcsGraph_setState(controller->getGraph(), observedGraph->q, observedGraph->q_dot);
    // compute from action model
    state[0] = this->controller->computeManipulabilityCost();
}

std::vector<std::string> Rcs::OMManipulabilityIndex::getStateNames() const
{
    return {"ManipIdx"};
}
