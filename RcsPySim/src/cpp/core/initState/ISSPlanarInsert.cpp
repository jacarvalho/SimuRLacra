#include "ISSPlanarInsert.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSPlanarInsert::ISSPlanarInsert(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    link1 = RcsGraph_getBodyByName(graph, "Link1");
    link2 = RcsGraph_getBodyByName(graph, "Link2");
    link3 = RcsGraph_getBodyByName(graph, "Link3");
    link4 = RcsGraph_getBodyByName(graph, "Link4");
    link5 = RcsGraph_getBodyByName(graph, "Link5");
    RCHECK(link1);
    RCHECK(link2);
    RCHECK(link3);
    RCHECK(link4);
    RCHECK(link5);
}

ISSPlanarInsert::~ISSPlanarInsert()
{
    // Nothing to destroy
}

unsigned int ISSPlanarInsert::getDim() const
{
    return 5;
}

void ISSPlanarInsert::getMinMax(double* min, double* max) const
{
    // Joint angles in rad (velocity stays on default)
    min[0] = -60./180*M_PI; // must enclose the init config coming from Pyrado
    max[0] = +60./180*M_PI; // must enclose the init config coming from Pyrado
    min[1] = -60./180*M_PI; // must enclose the init config coming from Pyrado
    max[1] = +60./180*M_PI; // must enclose the init config coming from Pyrado
    min[2] = -60./180*M_PI; // must enclose the init config coming from Pyrado
    max[2] = +60./180*M_PI; // must enclose the init config coming from Pyrado
    min[3] = -60./180*M_PI; // must enclose the init config coming from Pyrado
    max[3] = +60./180*M_PI; // must enclose the init config coming from Pyrado
    min[4] = -60./180*M_PI; // must enclose the init config coming from Pyrado
    max[4] = +60./180*M_PI; // must enclose the init config coming from Pyrado
}

std::vector<std::string> ISSPlanarInsert::getNames() const
{
    return {"q1", "q2", "q3", "q4", "q5"};
}

void ISSPlanarInsert::applyInitialState(const MatNd* initialState)
{
    // Get the relative joint angles
    double q1_init = initialState->ele[0];
    double q2_init = initialState->ele[1];
    double q3_init = initialState->ele[2];
    double q4_init = initialState->ele[3];
    double q5_init = initialState->ele[4];
    
    // Set the position to the links's rigid body joints
    graph->q->ele[link1->jnt->jointIndex] = q1_init;
    graph->q->ele[link2->jnt->jointIndex] = q2_init;
    graph->q->ele[link3->jnt->jointIndex] = q3_init;
    graph->q->ele[link4->jnt->jointIndex] = q4_init;
    graph->q->ele[link5->jnt->jointIndex] = q5_init;
    
    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}
    
} /* namespace Rcs */
