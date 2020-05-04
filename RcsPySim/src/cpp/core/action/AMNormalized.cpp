#include "AMNormalized.h"

#include <Rcs_VecNd.h>
#include <Rcs_macros.h>

namespace Rcs
{

AMNormalized::AMNormalized(ActionModel* wrapped) :
        ActionModel(wrapped->getGraph()), wrapped(wrapped)
{
    // Compute scale and shift from inner model bounds
    const MatNd* iModMin = wrapped->getSpace()->getMin();
    const MatNd* iModMax = wrapped->getSpace()->getMax();

    // shift is selected so that the median of min and max is 0
    // shift = min + (max - min)/2
    shift = MatNd_clone(iModMax);
    MatNd_subSelf(shift, iModMin);
    MatNd_constMulSelf(shift, 0.5);
    MatNd_addSelf(shift, iModMin);

    // scale = (max - min)/2
    scale = MatNd_clone(iModMax);
    MatNd_subSelf(scale, iModMin);
    MatNd_constMulSelf(scale, 0.5);
}

AMNormalized::~AMNormalized()
{
    delete wrapped;
}

unsigned int AMNormalized::getDim() const
{
    return wrapped->getDim();
}

void AMNormalized::getMinMax(double* min, double* max) const
{
    VecNd_setElementsTo(min, -1, getDim());
    VecNd_setElementsTo(max, 1, getDim());
}

std::vector<std::string> AMNormalized::getNames() const
{
    return wrapped->getNames();
}

void AMNormalized::computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des,
        const MatNd* action, double dt)
{
    // use temp storage for denormalized action, so that action remains unchanged.
    MatNd* denormalizedAction = NULL;
    MatNd_create2(denormalizedAction, action->m, action->n);
    // denormalize: denAction = action * scale + shift
    MatNd_eleMul(denormalizedAction, action, scale);
    MatNd_addSelf(denormalizedAction, shift);

    MatNd_maxSelf(denormalizedAction, wrapped->getSpace()->getMin());
    MatNd_minSelf(denormalizedAction, wrapped->getSpace()->getMax());

    //MatNd_printTranspose(denormalizedAction);

    // call wrapper
    wrapped->computeCommand(q_des, q_dot_des, T_des, denormalizedAction, dt);
    // destroy temp storage
    MatNd_destroy(denormalizedAction);
}

void AMNormalized::reset()
{
    wrapped->reset();
}

void AMNormalized::getStableAction(MatNd* action) const
{
    // compute wrapped stable action
    wrapped->getStableAction(action);
    // and normalize it
    for (unsigned int i = 0; i < getDim(); ++i) {
        action->ele[i] = (action->ele[i] - shift->ele[i]) / scale->ele[i];
    }
}

ActionModel* AMNormalized::getWrappedActionModel() const
{
    return wrapped;
}

ActionModel *AMNormalized::clone(RcsGraph *newGraph) const
{
    return new AMNormalized(wrapped->clone(newGraph));
}

} /* namespace Rcs */

