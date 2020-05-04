#include "ISSBoxLiftingSimple.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSBoxLiftingSimple::ISSBoxLiftingSimple(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    wrist1 = RcsGraph_getBodyByName(graph, "Wrist1");
    RCHECK(wrist1);
    wrist2 = RcsGraph_getBodyByName(graph, "Wrist2");
    RCHECK(wrist2);
    wrist3 = RcsGraph_getBodyByName(graph, "Wrist3");
    RCHECK(wrist3);
}

ISSBoxLiftingSimple::~ISSBoxLiftingSimple()
{
    // Nothing to destroy
}

unsigned int ISSBoxLiftingSimple::getDim() const
{
    return 3;
}

void ISSBoxLiftingSimple::getMinMax(double* min, double* max) const
{
    min[0] = 1.25;
    max[0] = 1.25;
    min[1] = -0.2;
    max[1] = -0.2;
    min[2] = 0.95;
    max[2] = 0.95;
}

std::vector<std::string> ISSBoxLiftingSimple::getNames() const
{
    return {"x", "y", "z"};
}

void ISSBoxLiftingSimple::applyInitialState(const MatNd* initialState)
{
    // Set the position to the box' rigid body joints directly in global world coordinates
    graph->q->ele[wrist1->jnt->jointIndex] = initialState->ele[0];
    graph->q->ele[wrist2->jnt->jointIndex] = initialState->ele[1];
    graph->q->ele[wrist3->jnt->jointIndex] = initialState->ele[2];

    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */
