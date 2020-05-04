#include "ExperimentConfig.h"
#include "action/AMJointControlPosition.h"
#include "action/AMIntegrate2ndOrder.h"
#include "action/AMTaskActivation.h"
#include "action/ActionModelIK.h"
#include "initState/ISSPlanar3Link.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMCombined.h"
#include "observation/OMJointState.h"
#include "observation/OMPartial.h"
#include "observation/OMGoalDistance.h"
#include "observation/OMForceTorque.h"
#include "observation/OMCollisionCost.h"
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

class ECPlanar3Link : public ExperimentConfig
{

protected:
    virtual ActionModel* createActionModel()
    {
        std::string actionModelType = "joint_pos";
        properties->getProperty(actionModelType, "actionModelType");
        
        if (actionModelType == "joint_pos")
        {
            return new AMJointControlPosition(graph);
        }
        else if (actionModelType == "joint_acc")
        {
            double max_action = 120*M_PI/180; // [1/s^2]
            return new AMIntegrate2ndOrder(new AMJointControlPosition(graph), max_action);
        }
        else if (actionModelType == "activation")
        {
            // Obtain the inner action model
            RcsBody* effector = RcsGraph_getBodyByName(graph, "Effector");
            RCHECK(effector);
            std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
            
            // Check if the MPs are defined on position or task level
            if (properties->getPropertyBool("positionTasks", true))
            {
                innerAM->addTask(new TaskPosition1D("X", graph, effector, nullptr, nullptr));
                innerAM->addTask(new TaskPosition1D("Z", graph, effector, nullptr, nullptr));
            }
            else
            {
                innerAM->addTask(new TaskVelocity1D("Xd", graph, effector, nullptr, nullptr));
                innerAM->addTask(new TaskVelocity1D("Zd", graph, effector, nullptr, nullptr));
            }
            
            // Obtain the task data
            auto& taskSpec = properties->getChildList("tasks");
            std::vector<std::unique_ptr<DynamicalSystem>> tasks;
            for (auto ts : taskSpec)
            {
                // All tasks cover the x and the z coordinate, thus no DSSlice is necessary
                tasks.emplace_back(DynamicalSystem::create(ts, innerAM->getDim()));
            }
            if (tasks.empty())
            {
                throw std::invalid_argument("No tasks specified!");
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

        if (properties->getPropertyBool("observeVelocities", true))
        {
            // Observe effector position and velocities
            auto omLin = new OMBodyStateLinear(graph, "Effector");  // in world coordinates
            omLin->setMinState(-1.56); // [m]
            omLin->setMaxState(1.56); // [m]
            omLin->setMaxVelocity(3.0); // [m/s]
            fullState->addPart(OMPartial::fromMask(omLin, {true, false, true}));  // mask out y axis
        }
        else
        {
            auto omLin = new OMBodyStateLinearPositions(graph, "Effector"); // in world coordinates
            omLin->setMinState(-1.56); // [m]
            omLin->setMaxState(1.56); // [m]
            fullState->addPart(OMPartial::fromMask(omLin, {true, false, true}));
        }

        std::string actionModelType = "joint_pos";
        properties->getProperty(actionModelType, "actionModelType");
        bool haveJointPos = actionModelType == "joint_pos";
        if (haveJointPos)
        {
            fullState->addPart(OMJointState::observeUnconstrainedJoints(graph));
        }
        else if (actionModelType == "activation")
        {
            if (properties->getPropertyBool("observeGoalDistance", false))
            {
                // Add goal distances
                auto castedAM = actionModel->unwrap<AMTaskActivation>();
                if (castedAM)
                {
                    auto omGoalDist = new OMGoalDistance(castedAM);
                    fullState->addPart(omGoalDist);
                }
                else
                {
                    delete fullState;
                    std::ostringstream os;
                    os << "The action model needs to be of type AMTaskActivation but is: " << castedAM;
                    throw std::invalid_argument(os.str());
                }
            }
            
            if (properties->getPropertyBool("observeDynamicalSystemDiscrepancy", false))
            {
                // Add the discrepancies between commanded and executed the task space changes
                auto castedAM = dynamic_cast<AMTaskActivation*>(actionModel);
                if (castedAM)
                {
                    auto omDescr = new OMDynamicalSystemDiscrepancy(castedAM);
                    fullState->addPart(omDescr);
                }
                else
                {
                    delete fullState;
                    std::ostringstream os;
                    os << "The action model needs to be of type AMTaskActivation but is: " << castedAM;
                    throw std::invalid_argument(os.str());
                }
            }
        }
        
        // Add force/torque measurements
        if (properties->getPropertyBool("observeForceTorque", true))
        {
            RcsSensor* fts = RcsGraph_getSensorByName(graph, "EffectorLoadCell");
            if (fts)
            {
                auto omForceTorque = new OMForceTorque(graph, fts->name, 500);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {true, false, true, false, false, false}));
            }
        }
        
        // Add current collision cost
        if (properties->getPropertyBool("observeCollisionCost", false) & (collisionMdl != nullptr))
        {
            // Add the collision cost observation model
            auto omColl = new OMCollisionCost(collisionMdl);
            fullState->addPart(omColl);
        }
        
        // Add collision prediction
        if (properties->getPropertyBool("observePredictedCollisionCost", false) && collisionMdl != nullptr)
        {
            // Get horizon from config
            int horizon = 20;
            properties->getChild("collisionConfig")->getProperty(horizon, "predCollHorizon");
            // Add collision model
            auto omCollisionCost = new OMCollisionCostPrediction(graph, collisionMdl, actionModel, 50);
            fullState->addPart(omCollisionCost);
        }
        
        // Add manipulability index
        auto ikModel = actionModel->unwrap<ActionModelIK>();
        if (properties->getPropertyBool("observeManipulabilityIndex", false) && ikModel)
        {
            bool ocm = properties->getPropertyBool("observeCurrentManipulability", true);
            fullState->addPart(new OMManipulabilityIndex(ikModel, ocm));
        }
        
        return fullState;
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Link1", new PPDMassProperties());
        manager->addParam("Link2", new PPDMassProperties());
        manager->addParam("Link3", new PPDMassProperties());
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSPlanar3Link(graph);
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* link3 = RcsGraph_getBodyByName(graph, "Link3");
        RCHECK(link3);
//        RcsBody* base = RcsGraph_getBodyByName(graph, "Base");
//        RCHECK(base);
//        return new ForceDisturber(link3, base);
        return new ForceDisturber(link3, link3);
    }
    
    virtual void initViewer(Rcs::Viewer* viewer)
    {
#ifdef GRAPHICS_AVAILABLE
        // Set camera next to base
        RcsBody* base = RcsGraph_getBodyByName(graph, "Base");
        double cameraCenter[3];
        Vec3d_copy(cameraCenter, base->A_BI->org);
        cameraCenter[1] -= 0.5;
        cameraCenter[2] += 0.3;
        
        // Set the camera position
        double cameraLocation[3];
        cameraLocation[0] = 0.;
        cameraLocation[1] = 4.;
        cameraLocation[2] = 2.5;
        
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

        auto omLin = observationModel->findOffsets<OMBodyStateLinear>();
        auto omLinPos = observationModel->findOffsets<OMBodyStateLinearPositions>();
        if (omLin)
        {
        linesOut.emplace_back(
            string_format("end-eff pos:   [% 1.3f,% 1.3f] m  end-eff vel:   [% 1.2f,% 1.2f] m/s",
                          obs->ele[omLin.pos], obs->ele[omLin.pos + 1],
                          obs->ele[sd + omLin.vel], obs->ele[sd + omLin.vel + 1]));
        }
        else if (omLinPos)
        {
            linesOut.emplace_back(
                string_format("end-eff pos:   [% 1.3f,% 1.3f] m",
                              obs->ele[omLin.pos], obs->ele[omLin.pos + 1]));
        }
        
        auto omGD = observationModel->findOffsets<OMGoalDistance>();
        if (omGD)
        {
            linesOut.emplace_back(
                string_format("goal dist pos: [% 1.3f,% 1.3f,% 1.3f] m",
                    obs->ele[omGD.pos + 0], obs->ele[omGD.pos + 1], obs->ele[omGD.pos + 2]));
        }
        
        auto omFT = observationModel->findOffsets<OMForceTorque>();
        if (omFT)
        {
            linesOut.emplace_back(
                string_format("forces:        [% 3.1f,% 3.1f] N", obs->ele[omFT.pos + 0], obs->ele[omFT.pos + 1]));
        }

        const double* distForce = forceDisturber->getLastForce();
        linesOut.emplace_back(
            string_format("disturbances:  [% 3.1f,% 3.1f,% 3.1f] N", distForce[0], distForce[1], distForce[2]));


        linesOut.emplace_back(
            string_format("actions:       [% 1.3f,% 1.3f,% 1.3f]",
                currentAction->ele[0], currentAction->ele[1], currentAction->ele[2]));
        
        if (physicsManager != nullptr)
        {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* link1_bpi = physicsManager->getBodyInfo("Link1");
            BodyParamInfo* link2_bpi = physicsManager->getBodyInfo("Link2");
            BodyParamInfo* link3_bpi = physicsManager->getBodyInfo("Link3");
            
            linesOut.emplace_back(
                string_format("link masses:   [% 1.3f,% 1.3f,% 1.3f] kg", link1_bpi->body->m, link2_bpi->body->m,
                              link3_bpi->body->m));
        }
        
        auto omManip = observationModel->findOffsets<OMManipulabilityIndex>();
        if (omManip)
        {
            linesOut.emplace_back(string_format("manipulability: % 5.3f", obs->ele[omManip.pos + 0]));
        }
    }
};

// Register
static ExperimentConfigRegistration<ECPlanar3Link> RegPlanar3Link("Planar3Link");

}
