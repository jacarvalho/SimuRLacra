#ifndef _AMJOINTCONTROL_H_
#define _AMJOINTCONTROL_H_

#include "ActionModel.h"

namespace Rcs
{

/**
 * Base class for all action models controlling the unconstrained joints of the graph directly (no IK).
 */
    class AMJointControl : public ActionModel
    {
    public:
        /**
         * Constructor
         * @param graph graph being commanded
         */
        explicit AMJointControl(RcsGraph* graph);
        
        virtual ~AMJointControl();
        
        /**
         * Returns graph->nJ since we control all unconstrained joints.
         */
        virtual unsigned int getDim() const;
    };
    
} /* namespace Rcs */

#endif /* _AMJOINTCONTROL_H_ */
