#include "ISSBallInTube.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSBallInTube::ISSBallInTube(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    platform = RcsGraph_getBodyByName(graph, "ImetronPlatform");
    RCHECK(platform);
    rail = RcsGraph_getBodyByName(graph, "RailBot");
    RCHECK(rail);
}

ISSBallInTube::~ISSBallInTube()
{
    // Nothing to destroy
}

unsigned int ISSBallInTube::getDim() const
{
    return 4;  // 3 base, 1 rail
}

void ISSBallInTube::getMinMax(double* min, double* max) const
{
    min[0] = -0.2;  // base_x
    max[0] = -0.0;
    min[1] = -0.05;  // base_y
    max[1] = 0.05;
    min[2] = RCS_DEG2RAD(-5.);  // base_theta
    max[2] = RCS_DEG2RAD(5.);
    min[3] = 0.8; // rail_z
    max[3] = 0.9;
}

std::vector<std::string> ISSBallInTube::getNames() const
{
    return {"base_x", "base_y", "base_theta", "rail_z"};
}

void ISSBallInTube::applyInitialState(const MatNd* initialState)
{
    // Set the position to the box' rigid body joints directly in global world coordinates
    bool b0 = RcsGraph_setJoint(graph, "DofBaseX", initialState->ele[0]);
    bool b1 = RcsGraph_setJoint(graph, "DofBaseY", initialState->ele[1]);
    bool b2 = RcsGraph_setJoint(graph, "DofBaseThZ", initialState->ele[2]);
    bool b3 = RcsGraph_setJoint(graph, "DofChestZ", initialState->ele[3]);
    if (!(b0 && b1 && b2 && b3))
    {
        throw std::invalid_argument("Setting graph failed for at least one of the joints!");
    }

    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */
