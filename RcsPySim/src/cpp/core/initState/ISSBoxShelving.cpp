#include "ISSBoxShelving.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSBoxShelving::ISSBoxShelving(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    platform = RcsGraph_getBodyByName(graph, "ImetronPlatform");
    RCHECK(platform);
    rail = RcsGraph_getBodyByName(graph, "RailBot");
    RCHECK(rail);
    link2L = RcsGraph_getBodyByName(graph, "lbr_link_2_L");
    RCHECK(link2L);
    link4L = RcsGraph_getBodyByName(graph, "lbr_link_4_L");
    RCHECK(link4L);
}

ISSBoxShelving::~ISSBoxShelving()
{
    // Nothing to destroy
}

unsigned int ISSBoxShelving::getDim() const
{
    return 6;  // 3 base, 1 rail, 2 LBR joints
}

void ISSBoxShelving::getMinMax(double* min, double* max) const
{
    min[0] = -0.1;  // base X
    max[0] = 0.1;
    min[1] = -0.1;  // base y
    max[1] = 0.1;
    min[2] = RCS_DEG2RAD(-10.);  // base theta z
    max[2] = RCS_DEG2RAD(10.);
    min[3] = 0.7; // rail z
    max[3] = 0.9;
    min[4] = RCS_DEG2RAD(20.); // joint 2
    max[4] = RCS_DEG2RAD(60.);
    min[5] = RCS_DEG2RAD(70.); // joint 4
    max[5] = RCS_DEG2RAD(95.);
}

std::vector<std::string> ISSBoxShelving::getNames() const
{
    return {"base_x", "base_y", "base_theta", "rail_z", "joint_2_L", "joint_4_L"};
}

void ISSBoxShelving::applyInitialState(const MatNd* initialState)
{
    // Set the position to the box' rigid body joints directly in global world coordinates
    bool b0 = RcsGraph_setJoint(graph, "DofBaseX", initialState->ele[0]);
    bool b1 = RcsGraph_setJoint(graph, "DofBaseY", initialState->ele[1]);
    bool b2 = RcsGraph_setJoint(graph, "DofBaseThZ", initialState->ele[2]);
    bool b3 = RcsGraph_setJoint(graph, "DofChestZ", initialState->ele[3]);
    bool b4 = RcsGraph_setJoint(graph, "lbr_joint_2_L", initialState->ele[4]);
    bool b5 = RcsGraph_setJoint(graph, "lbr_joint_4_L", initialState->ele[5]);
    if (!(b0 && b1 && b2 && b3 && b4 && b5))
    {
        throw std::invalid_argument("Setting graph failed for at least one of the joints!");
    }
    // Update the forward kinematics
    
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */
