#include "AMJointControlPosition.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>

#include <iostream>

namespace Rcs
{


AMJointControlPosition::AMJointControlPosition(RcsGraph* graph) : AMJointControl(graph)
{
    // Make sure nJ is correct
    RcsGraph_setState(graph, NULL, NULL);
    // Iterate over unconstrained joints
    REXEC(1)
    {
        RCSGRAPH_TRAVERSE_JOINTS(graph)
        {
            if (JNT->jacobiIndex != -1)
            {
                // Check if the joints actually use position control inside the simulation
                if (JNT->ctrlType != RCSJOINT_CTRL_POSITION)
                {
                    std::cout << "Using AMJointControlPosition, but at least one joint does not have the control type"
                                 "RCSJOINT_CTRL_POSITION!" << std::endl;
                }
            }
        }
    }
}

AMJointControlPosition::~AMJointControlPosition()
{
    // Nothing to destroy
}

void
AMJointControlPosition::computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt)
{
    RcsGraph_stateVectorFromIK(graph, action, q_des);
}

void AMJointControlPosition::getMinMax(double* min, double* max) const
{
    RCSGRAPH_TRAVERSE_JOINTS(graph)
    {
        if (JNT->jacobiIndex != -1)
        {
            // Set min/max from joint limits
            min[JNT->jacobiIndex] = JNT->q_min;
            max[JNT->jacobiIndex] = JNT->q_max;
        }
    }
}

void AMJointControlPosition::getStableAction(MatNd* action) const
{
    // Stable action = current state
    RcsGraph_stateVectorToIK(graph, graph->q, action);
}

std::vector<std::string> AMJointControlPosition::getNames() const
{
    std::vector<std::string> out;
    RCSGRAPH_TRAVERSE_JOINTS(graph)
    {
        if (JNT->jacobiIndex != -1)
        {
            out.emplace_back(JNT->name);
        }
    }

    return out;
}

ActionModel* AMJointControlPosition::clone(RcsGraph* newGraph) const
{
    return new AMJointControlPosition(newGraph);
}

} /* namespace Rcs */

