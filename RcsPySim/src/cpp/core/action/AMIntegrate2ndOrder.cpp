#include "AMIntegrate2ndOrder.h"

#include "../util/integrator.h"

#include <Rcs_VecNd.h>
#include <Rcs_macros.h>


namespace Rcs
{

AMIntegrate2ndOrder::AMIntegrate2ndOrder(ActionModel* wrapped, double maxAction) :
        ActionModel(wrapped->getGraph()),
        wrapped(wrapped)
{
    // Create integrator state storage
    integrated_action = MatNd_create(wrapped->getDim(), 1);
    integrated_action_dot = MatNd_create(wrapped->getDim(), 1);

    // Store max_action
    this->maxAction = MatNd_create(wrapped->getDim(), 1);
    VecNd_setElementsTo(this->maxAction->ele, maxAction, wrapped->getDim());
}

AMIntegrate2ndOrder::AMIntegrate2ndOrder(ActionModel* wrapped, MatNd* maxAction) :
                ActionModel(wrapped->getGraph()),
                wrapped(wrapped)
{
    // create integrator state storage
    integrated_action = MatNd_create(wrapped->getDim(), 1);
    integrated_action_dot = MatNd_create(wrapped->getDim(), 1);

    // store max_action
    RCHECK_MSG(maxAction->m == wrapped->getDim() && maxAction->n == 1, "MaxAction shape must match action dim.");
    this->maxAction = MatNd_clone(maxAction);
}


AMIntegrate2ndOrder::~AMIntegrate2ndOrder()
{
    MatNd_destroy(integrated_action_dot);
    MatNd_destroy(integrated_action);
    MatNd_destroy(maxAction);
    delete wrapped;
}

unsigned int AMIntegrate2ndOrder::getDim() const
{
    return wrapped->getDim();
}

void AMIntegrate2ndOrder::getMinMax(double* min, double* max) const
{
    VecNd_constMul(min, maxAction->ele, -1, getDim());
    VecNd_copy(max, maxAction->ele, getDim());
}

std::vector<std::string> AMIntegrate2ndOrder::getNames() const
{
    auto wnames = wrapped->getNames();
    // add the suffix 'dd' to every var to signal that it's the second order derivative
    for (auto& name : wnames) {
        name += "dd";
    }
    return wnames;
}

void AMIntegrate2ndOrder::computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt)
{
    // perform integration
    intStep2ndOrder(integrated_action, integrated_action_dot, action, dt, IntMode::ForwardEuler);

    // pass integrated values to wrapped
    wrapped->computeCommand(q_des, q_dot_des, T_des, integrated_action, dt);
}

void AMIntegrate2ndOrder::reset()
{
    wrapped->reset();
    // reset integrator state to current (initial)
    wrapped->getStableAction(integrated_action);
    MatNd_setZero(integrated_action_dot);
}

void AMIntegrate2ndOrder::getStableAction(MatNd* action) const
{
    // acceleration of 0 is stable
    MatNd_setZero(action);
}

ActionModel* AMIntegrate2ndOrder::getWrappedActionModel() const
{
    return wrapped;
}

ActionModel *AMIntegrate2ndOrder::clone(RcsGraph *newGraph) const
{
    return new AMIntegrate2ndOrder(wrapped->clone(newGraph), maxAction->ele[0]);
}

} /* namespace Rcs */
