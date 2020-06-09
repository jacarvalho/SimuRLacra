#include "ISSBoxLifting.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSBoxLifting::ISSBoxLifting(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    platform = RcsGraph_getBodyByName(graph, "ImetronPlatform");
    RCHECK(platform);
    rail = RcsGraph_getBodyByName(graph, "RailBot");
    RCHECK(rail);
    link2L = RcsGraph_getBodyByName(graph, "lbr_link_2_L");
    RCHECK(link2L);
    link2R = RcsGraph_getBodyByName(graph, "lbr_link_2_R");
    RCHECK(link2R);
}

ISSBoxLifting::~ISSBoxLifting()
{
    // Nothing to destroy
}

unsigned int ISSBoxLifting::getDim() const
{
    return 6;  // 3 base, 1 rail, 2 LBR joints
}

void ISSBoxLifting::getMinMax(double* min, double* max) const
{
    min[0] = 0.05;  // base_x
    max[0] = 0.25;
    min[1] = -0.05;  // base_y
    max[1] = 0.05;
    min[2] = RCS_DEG2RAD(-5.);  // base_theta
    max[2] = RCS_DEG2RAD(5.);
    min[3] = 0.8; // rail_z
    max[3] = 0.9;
    min[4] = RCS_DEG2RAD(60.); // joint_2_L
    max[4] = RCS_DEG2RAD(70.);
    min[5] = RCS_DEG2RAD(-70.); // joint_2_R
    max[5] = RCS_DEG2RAD(-60.);
}

std::vector<std::string> ISSBoxLifting::getNames() const
{
    return {"base_x", "base_y", "base_theta", "rail_z", "joint_2_L", "joint_2_R"};
}

void ISSBoxLifting::applyInitialState(const MatNd* initialState)
{
    // Set the position to the box' rigid body joints directly in global world coordinates
    bool b0 = RcsGraph_setJoint(graph, "DofBaseX", initialState->ele[0]);
    bool b1 = RcsGraph_setJoint(graph, "DofBaseY", initialState->ele[1]);
    bool b2 = RcsGraph_setJoint(graph, "DofBaseThZ", initialState->ele[2]);
    bool b3 = RcsGraph_setJoint(graph, "DofChestZ", initialState->ele[3]);
    bool b4 = RcsGraph_setJoint(graph, "lbr_joint_2_L", initialState->ele[4]);
    bool b5 = RcsGraph_setJoint(graph, "lbr_joint_2_R", initialState->ele[5]);
    if (!(b0 && b1 && b2 && b3 && b4 && b5))
    {
        throw std::invalid_argument("Setting graph failed for at least one of the joints!");
    }

    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */
