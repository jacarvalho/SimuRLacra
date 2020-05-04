#include "ISSMPBlending.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{
    
    ISSMPBlending::ISSMPBlending(RcsGraph* graph) : InitStateSetter(graph)
    {
        // Grab direct references to the body
        effector = RcsGraph_getBodyByName(graph, "Effector");
        RCHECK(effector);
    }
    
    ISSMPBlending::~ISSMPBlending()
    {
        // Nothing to destroy
    }
    
    unsigned int ISSMPBlending::getDim() const
    {
        return 2;
    }
    
    void ISSMPBlending::getMinMax(double* min, double* max) const
    {
        // Cartesian positions [m]
        min[0] = -1.;
        max[0] = +1.;
        min[1] = -1.;
        max[1] = +1.;
    }
    
    std::vector<std::string> ISSMPBlending::getNames() const
    {
        return {"x", "y"};
    }
    
    void ISSMPBlending::applyInitialState(const MatNd* initialState)
    {
        // Set the position to the links's rigid body joints
        graph->q->ele[effector->jnt->jointIndex] = initialState->ele[0];
        graph->q->ele[effector->jnt->jointIndex + 1] = initialState->ele[1];
        
        // Update the forward kinematics
        RcsGraph_setState(graph, graph->q, graph->q_dot);
    }
    
} /* namespace Rcs */
