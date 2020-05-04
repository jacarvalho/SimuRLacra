#include "ExperimentConfig.h"
#include "action/AMJointControlPosition.h"
#include "action/AMIntegrate2ndOrder.h"
#include "action/AMTaskActivation.h"
#include "action/ActionModelIK.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMBodyStateAngular.h"
#include "observation/OMCombined.h"
#include "observation/OMJointState.h"
#include "observation/OMPartial.h"
#include "observation/OMGoalDistance.h"
#include "observation/OMForceTorque.h"
#include "observation/OMCollisionCostPrediction.h"
#include "observation/OMManipulabilityIndex.h"
#include "observation/OMDynamicalSystemDiscrepancy.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"

#include <Rcs_Mat3d.h>
#include <Rcs_Vec3d.h>
#include <Rcs_typedef.h>
#include <Rcs_macros.h>

#include <TaskPosition1D.h>
#include <TaskVelocity1D.h>
#include <TaskOmega1D.h>

#include <sstream>
#include <stdexcept>
#include <cmath>
#include <memory>

namespace Rcs
{
    
    class ECPlanarBoxFiddle : public ExperimentConfig
    {
    
    protected:
        virtual ActionModel* createActionModel();
        
        virtual ObservationModel* createObservationModel();
    };
    
    
    ActionModel* ECPlanarBoxFiddle::createActionModel()
    {
        AMIKGeneric* ik = new AMIKGeneric(graph);
        
        RcsBody* effector = RcsGraph_getBodyByName(graph, "Effector");
        RCHECK(effector);
        RcsBody* refBdy = RcsGraph_getBodyByName(graph, "Goal");
        RCHECK(refBdy);
        ik->addTask(new TaskVelocity1D("Xd", graph, effector, refBdy, nullptr));
        ik->addTask(new TaskVelocity1D("Zd", graph, effector, refBdy, nullptr));
        ik->addTask(new TaskOmega1D("Bd", graph, effector, refBdy, nullptr));
        
        return ik;
    }
    
    ObservationModel* ECPlanarBoxFiddle::createObservationModel()
    {
        auto fullState = new OMCombined();
        
        // Observe effector position
        auto omLin = new OMBodyStateLinear(graph, "Effector", "Goal");
        omLin->setMinState(-1e10);
        omLin->setMaxState(1e10);
        omLin->setMaxVelocity(1e10);
        fullState->addPart(OMPartial::fromMask(omLin, {true, false, true}));  // position x- and z-axis
        
        // Observe effector orientation
        auto omAng = new OMBodyStateAngular(graph, "Effector", "Goal");
        omAng->setMinState(-1e10);
        omAng->setMaxState(1e10);
        omAng->setMaxVelocity(1e10);
        fullState->addPart(OMPartial::fromMask(omAng, {false, true, false}));  // rotation about y-axis
        
        return fullState;
    }
    
    
    // Register
    static ExperimentConfigRegistration<ECPlanarBoxFiddle> PlanarBoxFiddle("PlanarBoxFiddle");
    
}
