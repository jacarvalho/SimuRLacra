#include "ExperimentConfig.h"
#include "action/AMIntegrate2ndOrder.h"
#include "action/AMTaskActivation.h"
#include "action/ActionModelIK.h"
#include "initState/ISSPlanarInsert.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMBodyStateAngular.h"
#include "observation/OMCombined.h"
#include "observation/OMJointState.h"
#include "observation/OMPartial.h"
#include "observation/OMDynamicalSystemGoalDistance.h"
#include "observation/OMForceTorque.h"
#include "observation/OMCollisionCost.h"
#include "observation/OMCollisionCostPrediction.h"
#include "observation/OMManipulabilityIndex.h"
#include "observation/OMDynamicalSystemDiscrepancy.h"
#include "observation/OMTaskSpaceDiscrepancy.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDBodyPosition.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"
#include "physics/PPDMaterialProperties.h"

#include <Rcs_Mat3d.h>
#include <Rcs_Vec3d.h>
#include <Rcs_typedef.h>
#include <Rcs_macros.h>

#include <TaskVelocity1D.h>
#include <TaskOmega1D.h>

#ifdef GRAPHICS_AVAILABLE
#include <RcsViewer.h>
#endif

#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <memory>

namespace Rcs
{

class ECPlanarInsert : public ExperimentConfig
{

protected:
    virtual ActionModel* createActionModel()
    {
        std::string actionModelType = "activation";
        properties->getProperty(actionModelType, "actionModelType");

        // Common for the action models
        RcsBody* effector = RcsGraph_getBodyByName(graph, "Effector");
        RCHECK(effector);

        if (actionModelType == "ik")
        {
            // Create the action model
            auto amIK = new AMIKGeneric(graph);

            // Define velocity level Rcs tasks
            amIK->addTask(new TaskVelocity1D("Xd", graph, effector, nullptr, nullptr));
            amIK->addTask(new TaskVelocity1D("Zd", graph, effector, nullptr, nullptr));
            amIK->addTask(new TaskOmega1D("Bd", graph, effector, nullptr, nullptr));

            return amIK;
        }
        else if (actionModelType == "activation")
        {
            // Obtain the inner action model
            std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
            innerAM->addTask(new TaskVelocity1D("Xd", graph, effector, nullptr, nullptr));
            innerAM->addTask(new TaskVelocity1D("Zd", graph, effector, nullptr, nullptr));
            innerAM->addTask(new TaskOmega1D("Bd", graph, effector, nullptr, nullptr));
            
            // Obtain task data
            unsigned int i = 0;
            std::vector<unsigned int> offsets{0, 0, 1, 1, 2, 2}; // depends on the order of the MPs coming from python
            std::vector<std::unique_ptr<DynamicalSystem>> tasks;
            auto& taskSpec = properties->getChildList("tasks");
            for (auto tsk : taskSpec)
            {
                // Positive and negative linear velocity tasks separately
                DynamicalSystem* ds = DynamicalSystem::create(tsk, 1);
                tasks.emplace_back(new DSSlice(ds, offsets[i], 1));
                i++;
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
            os << "Unsupported action model type: ";
            os << actionModelType;
            throw std::invalid_argument(os.str());
        }
    }

    virtual ObservationModel* createObservationModel()
    {
        auto fullState = new OMCombined();
        
        // Observe effector position
        auto omLin = new OMBodyStateLinear(graph, "Effector", "GroundPlane"); // Base center is above ground level
        omLin->setMinState(-0.2); // [m] applied to X and Z
        omLin->setMaxState(1.5); // [m] applied to X and Z
        omLin->setMaxVelocity(0.5); // [m/s]
        fullState->addPart(OMPartial::fromMask(omLin, {true, false, true}));  // mask out y axis

        auto omAng = new OMBodyStateAngular(graph, "Effector", "GroundPlane"); // Base center is above ground level
        omAng->setMaxVelocity(20.); // [rad/s]
        fullState->addPart(OMPartial::fromMask(omAng, {false, true, false}));  // only y axis

        std::string actionModelType = "activation";
        properties->getProperty(actionModelType, "actionModelType");
        if (actionModelType == "activation")
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
                auto omForceTorque = new OMForceTorque(graph, fts->name, 300);
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
        
        // Add predicted collision cost
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

        // Add the task space discrepancy observation model
        if (properties->getPropertyBool("observeTaskSpaceDiscrepancy", false))
        {
            auto wamIK = actionModel->unwrap<ActionModelIK>();
            if (wamIK)
            {
                auto omTSDescr = new OMTaskSpaceDiscrepancy("Effector", graph, wamIK->getController()->getGraph());
                fullState->addPart(OMPartial::fromMask(omTSDescr, {true, false, true}));
            }
            else
            {
                delete fullState;
                throw std::invalid_argument("The action model needs to be of type ActionModelIK!");
            }
        }
        
        return fullState;
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Link1", new PPDMassProperties());
        manager->addParam("Link2", new PPDMassProperties());
        manager->addParam("Link3", new PPDMassProperties());
        manager->addParam("Link4", new PPDMassProperties());
        manager->addParam("Effector", new PPDMaterialProperties());
        manager->addParam("UpperWall", new PPDBodyPosition(false, false, true)); // only z position
        manager->addParam("UpperWall", new PPDMaterialProperties());
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSPlanarInsert(graph);
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* effector = RcsGraph_getBodyByName(graph, "Effector");
        RCHECK(effector);
        return new ForceDisturber(effector, effector);
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
    getHUDText(std::vector<std::string>& linesOut, double currentTime, const MatNd* obs, const MatNd* currentAction,
               PhysicsBase* simulator, PhysicsParameterManager* physicsManager, ForceDisturber* forceDisturber) override
    {
        // Obtain simulator name
        const char* simName = "None";
        if (simulator != nullptr)
        {
            simName = simulator->getClassName();
        }
        
        linesOut.emplace_back(
            string_format("physics engine: %s                            sim time: %2.3f s", simName, currentTime));

        unsigned int sd = observationModel->getStateDim();

        auto omLin = observationModel->findOffsets<OMBodyStateLinear>();
        auto omAng = observationModel->findOffsets<OMBodyStateAngular>();
        if (omLin && omAng)
        {

            linesOut.emplace_back(string_format("end-eff pos:   [% 1.3f,% 1.3f,% 1.3f] m, m, deg",
                                  obs->ele[omLin.pos], obs->ele[omLin.pos + 1], RCS_RAD2DEG(obs->ele[omAng.pos])));

            linesOut.emplace_back(string_format("end-eff vel:   [% 1.3f,% 1.3f,% 1.3f] m/s, m/s, deg/s",
                                                obs->ele[sd + omLin.vel],
                                                obs->ele[sd + omLin.vel + 1],
                                                RCS_RAD2DEG(obs->ele[sd + omAng.vel])));
        }

        auto omFT = observationModel->findOffsets<OMForceTorque>();
        if (omFT)
        {
            linesOut.emplace_back(
                string_format("forces:        [% 3.1f,% 3.1f] N", obs->ele[omFT.pos], obs->ele[omFT.pos + 1]));
        }

        auto omTSD = observationModel->findOffsets<OMTaskSpaceDiscrepancy>();
        if (omTSD)
        {
            linesOut.emplace_back(
                string_format("ts delta:      [% 1.3f,% 1.3f] m", obs->ele[omTSD.pos], obs->ele[omTSD.pos + 1]));
        }
    
        std::stringstream ss;
        ss << "actions:       [";
        for (unsigned int i = 0; i < currentAction->m - 1; i++)
        {
            ss << std::fixed << std::setprecision(3) << MatNd_get(currentAction, i, 0) << ", ";
            if (i == 6)
            {
                ss << "\n               ";
            }
        }
        ss << std::fixed << std::setprecision(3) << MatNd_get(currentAction, currentAction->m - 1, 0) << "]";
        linesOut.emplace_back(string_format(ss.str()));
    
        auto castedAM = dynamic_cast<AMTaskActivation*>(actionModel);
        if (castedAM)
        {
            std::stringstream ss;
            ss << "activations:   [";
            for (unsigned int i = 0; i < castedAM->getDim() - 1; i++)
            {
                ss << std::fixed << std::setprecision(3) << MatNd_get(castedAM->getActivation(), i, 0) << ", ";
                if (i == 6)
                {
                    ss << "\n               ";
                }
            }
            ss << std::fixed << std::setprecision(3) <<
               MatNd_get(castedAM->getActivation(), castedAM->getDim() - 1, 0) << "]";
            linesOut.emplace_back(string_format(ss.str()));
        
            linesOut.emplace_back(string_format("tcm:            %s", castedAM->getTaskCombinationMethodName()));
        }
        
        const double* distForce = forceDisturber->getLastForce();
        linesOut.emplace_back(
            string_format("disturbances:  [% 3.1f,% 3.1f,% 3.1f] N", distForce[0], distForce[1], distForce[2]));
        
        if (physicsManager != nullptr)
        {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* link1_bpi = physicsManager->getBodyInfo("Link1");
            BodyParamInfo* link2_bpi = physicsManager->getBodyInfo("Link2");
            BodyParamInfo* link3_bpi = physicsManager->getBodyInfo("Link3");
            BodyParamInfo* link4_bpi = physicsManager->getBodyInfo("Link4");
            BodyParamInfo* uWall_bpi = physicsManager->getBodyInfo("UpperWall");
            BodyParamInfo* lWall_bpi = physicsManager->getBodyInfo("LowerWall");
            BodyParamInfo* eff_bpi = physicsManager->getBodyInfo("Effector");
            
            linesOut.emplace_back(
                string_format("link masses:   [%1.2f, %1.2f, %1.2f, %1.2f] kg      wall Z pos: %1.3f m",
                              link1_bpi->body->m, link2_bpi->body->m, link3_bpi->body->m, link4_bpi->body->m,
                              uWall_bpi->body->A_BP->org[2]));
            linesOut.emplace_back(string_format("wall friction: [%1.3f, %1.3f]            effector friction: %1.3f",
                                                lWall_bpi->material.getFrictionCoefficient(),
                                                uWall_bpi->material.getFrictionCoefficient(),
                                                eff_bpi->material.getFrictionCoefficient()));
        }
    }
};

// Register
static ExperimentConfigRegistration<ECPlanarInsert> RegPlanarInsert("PlanarInsert");
    
}
