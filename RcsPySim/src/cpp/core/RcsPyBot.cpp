#include "RcsPyBot.h"

#include "action/ActionModel.h"
#include "observation/ObservationModel.h"
#include "control/ControlPolicy.h"

#include "control/MLPPolicy.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>
#include <Rcs_timer.h>

#include <cmath>

namespace Rcs
{

//namespace {

/*! A simple time-based policy
 * Yields 2 actions: the first one is oscillating round 0 with am amplitude of 0.2 and the second one is constant 0.
 */
class SimpleControlPolicy: public ControlPolicy
{
private:
    //! Internal clock
    double t;

public:
    SimpleControlPolicy()
    { t = 0; }

    virtual void computeAction(MatNd* action, const MatNd* observation)
    {
        action->ele[0]= 0.2 * std::cos(2.*M_PI * t) * (135 * M_PI / 180);
        action->ele[1]= 0.0;
        t += 0.01;
    }
};

//}

RcsPyBot::RcsPyBot(PropertySource* propertySource)
{
    // Load experiment config
    config = ExperimentConfig::create(propertySource);

    // Set MotionControlLayer members
    currentGraph = config->graph;
    desiredGraph = RcsGraph_clone(currentGraph);

    // Control policy is set later
    controlPolicy = NULL;

    // Init temp matrices, making sure the initial command is identical to the initial state
    q_ctrl = MatNd_clone(desiredGraph->q);
    qd_ctrl = MatNd_clone(desiredGraph->q_dot);
    T_ctrl = MatNd_create(desiredGraph->dof, 1);

    observation = config->observationModel->getSpace()->createValueMatrix();
    action = config->actionModel->getSpace()->createValueMatrix();

    // ActionModel and observationModel expect a reset() call before they are used
    config->actionModel->reset();
    config->observationModel->reset();
}

RcsPyBot::~RcsPyBot()
{
    // Delete temporary matrices
    MatNd_destroy(q_ctrl);
    MatNd_destroy(qd_ctrl);
    MatNd_destroy(T_ctrl);

    MatNd_destroy(observation);
    MatNd_destroy(action);

    // The hardware components also use currentGraph, so it may only be destroyed by the MotionControlLayer destructor.
    // however, currentGraph is identical to config->graph, which is owned.
    // to solve this, set config->graph to NULL.
    config->graph = NULL;
    // Also, desiredGraph is a clone, so it must be destroyed. Can't set MotionControlLayer::ownsDesiredGraph = true
    // since it's private, so do it manually here.
    RcsGraph_destroy(desiredGraph);

    // Delete experiment config
    delete config;
}

void RcsPyBot::setControlPolicy(ControlPolicy* controlPolicy)
{
    std::unique_lock<std::mutex> lock(controlPolicyMutex);
    this->controlPolicy = controlPolicy;
    if (controlPolicy == NULL) {
        // command initial state
        RcsGraph_getDefaultState(desiredGraph, q_ctrl);
        MatNd_setZero(qd_ctrl);
        MatNd_setZero(T_ctrl);
    }
    // reset model states
    config->observationModel->reset();
    config->actionModel->reset();
}

void RcsPyBot::updateControl()
{
    // aggressive locking here is ok, setControlPolicy doesn't take long
    std::unique_lock<std::mutex> lock(controlPolicyMutex);
    // read observation from current graph
    config->observationModel->computeObservation(observation, action, config->dt);

    // compute action
    if (controlPolicy != NULL) {
        controlPolicy->computeAction(action, observation);

        // run action through action model
        config->actionModel->computeCommand(q_ctrl, qd_ctrl, T_ctrl, action,
                getCallbackUpdatePeriod());
    }
    // XXX TEST
    //MatNd_printTranspose(action);

    // update desired state graph

    // TODO if(writeCommands), but seriously, shouldn't the component handle that stuff itself?
    // command action to hardware (and update desiredGraph)
    setMotorCommand(q_ctrl, qd_ctrl, T_ctrl);

    // can unlock now, lock only guards controlPolicy and ctrl vectors
    lock.unlock();

    // log data
    double reward = 0.0; // TODO compute
    logger.record(observation, action, reward);
}

MatNd *RcsPyBot::getObservation() const {
    return observation;
}

MatNd *RcsPyBot::getAction() const {
    return action;
}


} /* namespace Rcs */
