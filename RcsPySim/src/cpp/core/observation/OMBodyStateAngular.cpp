#include "OMBodyStateAngular.h"

#include <TaskEuler3D.h>

namespace Rcs
{

OMBodyStateAngular::OMBodyStateAngular(
    RcsGraph* graph,
    const char* effectorName,
    const char* refBodyName, const char* refFrameName) :
    OMTask(new TaskEuler3D(graph, NULL, NULL, NULL))
{
    initTaskBodyNames(effectorName, refBodyName, refFrameName);
}

OMBodyStateAngularPositions::OMBodyStateAngularPositions(
    RcsGraph* graph,
    const char* effectorName,
    const char* refBodyName, const char* refFrameName) :
    OMTaskPositions(new TaskEuler3D(graph, NULL, NULL, NULL))
{
    initTaskBodyNames(effectorName, refBodyName, refFrameName);
}

} /* namespace Rcs */
