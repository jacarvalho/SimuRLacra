#include "ISSBoxFlipping.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSBoxFlipping::ISSBoxFlipping(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    wrist1L = RcsGraph_getBodyByName(graph, "Wrist1_L");
    RCHECK(wrist1L);
    wrist2L = RcsGraph_getBodyByName(graph, "Wrist2_L");
    RCHECK(wrist2L);
    wrist3L = RcsGraph_getBodyByName(graph, "Wrist3_L");
    RCHECK(wrist3L);
    wrist1R = RcsGraph_getBodyByName(graph, "Wrist1_R");
    RCHECK(wrist1R);
    wrist2R = RcsGraph_getBodyByName(graph, "Wrist2_R");
    RCHECK(wrist2R);
    wrist3R = RcsGraph_getBodyByName(graph, "Wrist3_R");
    RCHECK(wrist3R);
}

ISSBoxFlipping::~ISSBoxFlipping()
{
    // Nothing to destroy
}

unsigned int ISSBoxFlipping::getDim() const
{
    return 6;
}

void ISSBoxFlipping::getMinMax(double* min, double* max) const
{
    min[0] = 1.25;
    max[0] = 1.25;
    min[1] = 0.2;
    max[1] = 0.2;
    min[2] = 0.95;
    max[2] = 0.95;
    min[3] = 1.25;
    max[3] = 1.25;
    min[4] = -0.2;
    max[4] = -0.2;
    min[5] = 0.95;
    max[5] = 0.95;
}

std::vector<std::string> ISSBoxFlipping::getNames() const
{
    return {"x_L", "y_L", "z_L", "x_L", "y_L", "z_L"};
}

void ISSBoxFlipping::applyInitialState(const MatNd* initialState)
{
    // Set the position to the box' rigid body joints directly in global world coordinates
    graph->q->ele[wrist1L->jnt->jointIndex] = initialState->ele[0];
    graph->q->ele[wrist2L->jnt->jointIndex] = initialState->ele[1];
    graph->q->ele[wrist3L->jnt->jointIndex] = initialState->ele[2];
    graph->q->ele[wrist1R->jnt->jointIndex] = initialState->ele[3];
    graph->q->ele[wrist2R->jnt->jointIndex] = initialState->ele[4];
    graph->q->ele[wrist3R->jnt->jointIndex] = initialState->ele[5];

    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */
