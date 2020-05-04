#include "AMJointControl.h"

#include <Rcs_typedef.h>

namespace Rcs
{

AMJointControl::AMJointControl(RcsGraph* graph) : ActionModel(graph)
{
    // nothing to do here
}

AMJointControl::~AMJointControl()
{
    // nothing to destroy
}

unsigned int AMJointControl::getDim() const
{
    // all unconstrained joints
    return graph->nJ;
}

} /* namespace Rcs */

