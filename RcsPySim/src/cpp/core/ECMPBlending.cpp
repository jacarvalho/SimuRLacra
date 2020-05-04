#include "ExperimentConfig.h"
#include "action/AMJointControlPosition.h"
#include "action/AMIntegrate2ndOrder.h"
#include "action/AMTaskActivation.h"
#include "action/ActionModelIK.h"
#include "initState/ISSMPBlending.h"
#include "observation/OMBodyStateLinear.h"
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

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>

#endif

#include <sstream>
#include <stdexcept>
#include <cmath>
#include <memory>

namespace Rcs
{
    
    class ECMPblending : public ExperimentConfig
    {
    
    protected:
        virtual ActionModel* createActionModel()
        {
            std::string actionModelType = "UNSPECIFIED";
            properties->getProperty(actionModelType, "actionModelType");
    
            if (actionModelType == "activation")
            {
                // Obtain the inner action model
                RcsBody* effector = RcsGraph_getBodyByName(graph, "Effector");
                RCHECK(effector);
                std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
    
                // Check if the MPs are defined on position or task level
                if (properties->getPropertyBool("positionTasks", true))
                {
                    innerAM->addTask(new TaskPosition1D("X", graph, effector, nullptr, nullptr));
                    innerAM->addTask(new TaskPosition1D("Y", graph, effector, nullptr, nullptr));
                }
                else
                {
                    innerAM->addTask(new TaskVelocity1D("Xd", graph, effector, nullptr, nullptr));
                    innerAM->addTask(new TaskVelocity1D("Yd", graph, effector, nullptr, nullptr));
                }
    
                // Obtain the task data
                auto& taskSpec = properties->getChildList("tasks");
                if (taskSpec.empty())
                {
                    throw std::invalid_argument("No tasks specified!");
                }
                std::vector<std::unique_ptr<DynamicalSystem>> tasks;
                for (auto ts : taskSpec)
                {
                    // All tasks cover the x and y coordinate
                    tasks.emplace_back(DynamicalSystem::create(ts, innerAM->getDim()));
                }
    
                // Incorporate collision costs into IK
                if (properties->getPropertyBool("collisionAvoidanceIK", true))
                {
                    std::cout << "IK considers the provided collision model" << std::endl;
                    innerAM->setupCollisionModel(collisionMdl);
                }
    
                // Setup task-based action model
                std::vector<DynamicalSystem*> taskRel;
                for (auto& task : tasks)
                {
                    taskRel.push_back(task.release());
                }
    
                // Get the method how to combine the movement primitives / tasks given their activation
                std::string taskCombinationMethod = "mean";
                properties->getProperty(taskCombinationMethod, "taskCombinationMethod");
                TaskCombinationMethod tcm = AMTaskActivation::checkTaskCombinationMethod(taskCombinationMethod);
    
                // Create the action model
                return new AMTaskActivation(innerAM.release(), taskRel, tcm);
            }
            else
            {
                std::ostringstream os;
                os << "Unsupported action model type: " << actionModelType;
                throw std::invalid_argument(os.str());
            }
        }
        
        virtual ObservationModel* createObservationModel()
        {
            auto fullState = new OMCombined();
            
            // Observe effector position
            auto omLin = new OMBodyStateLinear(graph, "Effector", "GroundPlane");
            fullState->addPart(OMPartial::fromMask(omLin, {true, true, false}));  // mask out z axis
            
            return fullState;
        }
        
        virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
        {
            manager->addParam("Effector", new PPDMassProperties());  // not necessary
        }
    
    public:
        virtual InitStateSetter* createInitStateSetter()
        {
            return new ISSMPBlending(graph);
        }
        
        virtual ForceDisturber* createForceDisturber()
        {
            RcsBody* eff = RcsGraph_getBodyByName(graph, "Effector");
            RCHECK(eff);
            return new ForceDisturber(eff, NULL);
        }
        
        virtual void initViewer(Rcs::Viewer* viewer)
        {
#ifdef GRAPHICS_AVAILABLE
            // Set camera next to base
            RcsBody* base = RcsGraph_getBodyByName(graph, "Effector");
            double cameraCenter[3];
            Vec3d_copy(cameraCenter, base->A_BI->org);
            cameraCenter[1] -= 0.0;
            cameraCenter[2] += 0.0;
            
            // Set the camera position
            double cameraLocation[3];
            cameraLocation[0] = 0.;
            cameraLocation[1] = -2.5;
            cameraLocation[2] = 4.;
            
            // Camera up vector defaults to z
            double cameraUp[3];
            Vec3d_setUnitVector(cameraUp, 2);
            
            // Apply camera position
            viewer->setCameraHomePosition(osg::Vec3d(cameraLocation[0], cameraLocation[1], cameraLocation[2]),
                                          osg::Vec3d(cameraCenter[0], cameraCenter[1], cameraCenter[2]),
                                          osg::Vec3d(cameraUp[0], cameraUp[1], cameraUp[2]));
#endif
        }
        
        void
        getHUDText(
            std::vector<std::string>& linesOut,
            double currentTime,
            const MatNd* obs,
            const MatNd* currentAction,
            PhysicsBase* simulator,
            PhysicsParameterManager* physicsManager,
            ForceDisturber* forceDisturber) override
        {
            // Obtain simulator name
            const char* simName = "None";
            if (simulator != nullptr)
            {
                simName = simulator->getClassName();
            }
            
            linesOut.emplace_back(
            string_format("physics engine: %s                           sim time: %2.3f s", simName, currentTime));

            unsigned int sd = observationModel->getStateDim();

            linesOut.emplace_back(
            string_format("end-eff pos:   [% 1.3f,% 1.3f] m  end-eff vel:   [% 1.2f,% 1.2f] m/s",
                          obs->ele[0], obs->ele[1], obs->ele[sd], obs->ele[sd + 1]));
            
            linesOut.emplace_back(
            string_format("actions:       [% 1.3f,% 1.3f,% 1.3f,% 1.3f]", currentAction->ele[0], currentAction->ele[1],
                          currentAction->ele[2], currentAction->ele[3]));
            
            const double* distForce = forceDisturber->getLastForce();
            linesOut.emplace_back(
            string_format("disturbances:   [% 3.1f,% 3.1f,% 3.1f] N", distForce[0], distForce[1], distForce[2]));
            
            if (physicsManager != nullptr)
            {
                // Get the parameters that are not stored in the Rcs graph
                BodyParamInfo* eff_bpi = physicsManager->getBodyInfo("Effector");
                linesOut.emplace_back(string_format("effector mass:    % 1.3f kg", eff_bpi->body->m));
            }
        }
    };

// Register
    static ExperimentConfigRegistration<ECMPblending> RegMPBlending("MPBlending");
    
}
