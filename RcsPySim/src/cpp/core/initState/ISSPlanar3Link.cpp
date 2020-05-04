#include "ISSPlanar3Link.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{
    
    ISSPlanar3Link::ISSPlanar3Link(RcsGraph* graph) : InitStateSetter(graph)
    {
        // Grab direct references to the used bodies
        link1 = RcsGraph_getBodyByName(graph, "Link1");
        link2 = RcsGraph_getBodyByName(graph, "Link2");
        link3 = RcsGraph_getBodyByName(graph, "Link3");
        RCHECK(link1);
        RCHECK(link2);
        RCHECK(link3);
    }
    
    ISSPlanar3Link::~ISSPlanar3Link()
    {
        // Nothing to destroy
    }
    
    unsigned int ISSPlanar3Link::getDim() const
    {
        return 3;
    }
    
    void ISSPlanar3Link::getMinMax(double* min, double* max) const
    {
        // Joint angles [rad] (velocity stays on default)
        min[0] = -90./180*M_PI;
        max[0] = +90./180*M_PI;
        min[1] = -160./180*M_PI;
        max[1] = +160./180*M_PI;
        min[2] = -160./180*M_PI;
        max[2] = +160./180*M_PI;
    }
    
    std::vector<std::string> ISSPlanar3Link::getNames() const
    {
        return {"q1", "q2", "q3"};
    }
    
    void ISSPlanar3Link::applyInitialState(const MatNd* initialState)
    {
        // Get the relative joint angles
        double q1_init = initialState->ele[0];
        double q2_init = initialState->ele[1];
        double q3_init = initialState->ele[2];
        
        // Set the position to the links's rigid body joints
        graph->q->ele[link1->jnt->jointIndex] = q1_init;
        graph->q->ele[link2->jnt->jointIndex] = q2_init;
        graph->q->ele[link3->jnt->jointIndex] = q3_init;
        
        // Update the forward kinematics
        RcsGraph_setState(graph, graph->q, graph->q_dot);
    }
    
} /* namespace Rcs */
