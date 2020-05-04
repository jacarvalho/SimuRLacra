#include "OMBodyStateLinear.h"

#include <TaskPosition3D.h>

namespace Rcs
{

OMBodyStateLinear::OMBodyStateLinear(
    RcsGraph* graph,
    const char* effectorName,
    const char* refBodyName,
    const char* refFrameName) :
    OMTask(new TaskPosition3D(graph, NULL, NULL, NULL))
{
    initTaskBodyNames(effectorName, refBodyName, refFrameName);
}

OMBodyStateLinearPositions::OMBodyStateLinearPositions(
    RcsGraph* graph,
    const char* effectorName,
    const char* refBodyName, const char* refFrameName) :
    OMTaskPositions(new TaskPosition3D(graph, NULL, NULL, NULL))
{
    initTaskBodyNames(effectorName, refBodyName, refFrameName);
}

} /* namespace Rcs */

