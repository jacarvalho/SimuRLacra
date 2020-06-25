#include "OMDynamicalSystemGoalDistance.h"
#include "../action/AMTaskActivation.h"
#include "../action/ActionModelIK.h"
#include "../util/eigen_matnd.h"

#include <ControllerBase.h>
#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <Rcs_VecNd.h>

#include <algorithm>
#include <limits>

namespace Rcs
{

OMGoalDistance::OMGoalDistance(AMTaskActivation* actionModel) :
        actionModel(actionModel),
        maxDistance(std::numeric_limits<double>::infinity())
{
    auto amik = dynamic_cast<AMIKGeneric*>(actionModel->getWrappedActionModel());
    RCHECK_MSG(amik, "AMTaskActivation must wrap an AMIKGeneric");

    controller = new ControllerBase(actionModel->getGraph());
    for (auto tsk : amik->getController()->getTasks()) {
        controller->add(tsk->clone(actionModel->getGraph()));
    }
}

OMGoalDistance::~OMGoalDistance()
{
    delete controller;
}

unsigned int OMGoalDistance::getStateDim() const
{
    return actionModel->getDim();
}

unsigned int OMGoalDistance::getVelocityDim() const
{
    return 0;
}

void OMGoalDistance::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
{
    // Compute controller state
    Eigen::VectorXd x_curr = Eigen::VectorXd::Zero(controller->getTaskDim());
    MatNd x_curr_mat = viewEigen2MatNd(x_curr);
    controller->computeX(&x_curr_mat);

    // Compute goal distance derivative
    auto& tasks = actionModel->getDynamicalSystems();
    for (size_t i = 0; i < tasks.size(); ++i) {
        // Compute distance
        double dist = tasks[i]->goalDistance(x_curr);
        state[i] = dist;

        // DEBUG
        REXEC(7) {
            std::cout << "goal distance pos of task " << i << std::endl << state[i] << std::endl;
        }
    }
}

void OMGoalDistance::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    VecNd_setZero(minState, getStateDim()); // minimum distance is 0
    VecNd_setElementsTo(maxState, maxDistance, getStateDim());
}

std::vector<std::string> OMGoalDistance::getStateNames() const
{
    std::vector<std::string> result;
    result.reserve(getStateDim());
    for (size_t ds = 0; ds < actionModel->getDynamicalSystems().size(); ++ds) {
        std::ostringstream os;
        os << "GD_DS" << ds;
        result.push_back(os.str());
    }
    return result;
}


} /* namespace Rcs */

