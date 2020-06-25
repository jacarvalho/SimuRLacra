#include "ExperimentConfig.h"
#include "action/ActionModelIK.h"
#include "action/AMTaskActivation.h"
#include "observation/OMCombined.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMCollisionCost.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDSphereRadius.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "observation/OMDynamicalSystemGoalDistance.h"
#include "util/string_format.h"
#include "observation/OMCollisionCostPrediction.h"

#include <Rcs_macros.h>
#include <TaskPosition3D.h>

#ifdef GRAPHICS_AVAILABLE
#include <RcsViewer.h>
#endif

#include <memory>

namespace Rcs {

class ECTargetTracking : public ExperimentConfig
{

protected:
    virtual ActionModel* createActionModel()
    {
        // Setup inner action model
        RcsBody* left = RcsGraph_getBodyByName(graph, "PowerGrasp_L");
        RCHECK(left);
        RcsBody* right = RcsGraph_getBodyByName(graph, "PowerGrasp_R");
        RCHECK(right);

        // Control effector positions (not orientation)
        std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
        innerAM->addTask(new TaskPosition3D(graph, left, nullptr, nullptr));
        innerAM->addTask(new TaskPosition3D(graph, right, nullptr, nullptr));

        // Incorporate collision costs into IK
        if (properties->getPropertyBool("collisionAvoidanceIK", true))
        {
            std::cout << "IK considers the provided collision model" << std::endl;
            innerAM->setupCollisionModel(collisionMdl);
        }

        // Obtain task data
        std::vector<std::unique_ptr<DynamicalSystem>> tasks;
        auto& tsLeft = properties->getChildList("tasksLeft");
        for (auto tsk : tsLeft)
        {
            DynamicalSystem* ds = DynamicalSystem::create(tsk, 3);
            tasks.emplace_back(new DSSlice(ds, 0, 3));
        }
        auto& tsRight = properties->getChildList("tasksRight");
        for (auto tsk : tsRight)
        {
            DynamicalSystem* ds = DynamicalSystem::create(tsk, 3);
            tasks.emplace_back(new DSSlice(ds, 3, 3));
        }
        if (tasks.empty())
        {
            throw std::invalid_argument("No tasks specified!");
        }

        // Setup task-based action model
        std::vector<DynamicalSystem*> taskRel;
        for (auto& task : tasks) {
            taskRel.push_back(task.release());
        }
    
        // Get the method how to combine the movement primitives / tasks given their activation
        std::string taskCombinationMethod = "mean";
        properties->getProperty(taskCombinationMethod, "taskCombinationMethod");
        TaskCombinationMethod tcm = AMTaskActivation::checkTaskCombinationMethod(taskCombinationMethod);
        
        return new AMTaskActivation(innerAM.release(), taskRel, tcm);
    }

    virtual ObservationModel* createObservationModel()
    {
        // Observe effector positions
        std::unique_ptr<OMCombined> fullState(new OMCombined());

        auto left = new OMBodyStateLinear(graph, "PowerGrasp_L");
        fullState->addPart(left);

        auto right = new OMBodyStateLinear(graph, "PowerGrasp_R");
        fullState->addPart(right);

        auto amAct = actionModel->unwrap<AMTaskActivation>();
        RCHECK(amAct);
        fullState->addPart(new OMGoalDistance(amAct));

        if (properties->getPropertyBool("observeCollisionCost", true) & (collisionMdl != nullptr))
        {
            // Add the collision cost observation model
            auto omColl = new OMCollisionCost(collisionMdl);
            fullState->addPart(omColl);
        }
    
        if (properties->getPropertyBool("observePredictedCollisionCost", false) & (collisionMdl != nullptr))
        {
            // Get the horizon from config
            int horizon = 20;
            properties->getChild("collisionConfig")->getProperty(horizon, "predCollHorizon");
            // Add the collision cost observation model
            auto omCollPred = new OMCollisionCostPrediction(graph, collisionMdl, actionModel, horizon);
            fullState->addPart(omCollPred);
        }


        return fullState.release();
    }

public:
    void
    getHUDText(std::vector<std::string>& linesOut, double currentTime, const MatNd* obs, const MatNd* currentAction,
               PhysicsBase* simulator, PhysicsParameterManager* physicsManager, ForceDisturber* forceDisturber) override
    {
        // Obtain simulator name
        const char* simName = "None";
        if (simulator != nullptr) {
            simName = simulator->getClassName();
        }

        linesOut.emplace_back(string_format(
                "physics engine: %s        simulation time:             %2.3f s",
                simName, currentTime));

        linesOut.emplace_back(string_format(
                " left hand pg:     [% 3.2f,% 3.2f,% 3.2f] m",
                obs->ele[0], obs->ele[1], obs->ele[2]));

        linesOut.emplace_back(string_format(
                "right hand pg:     [% 3.2f,% 3.2f,% 3.2f] m",
                obs->ele[3], obs->ele[4], obs->ele[5]));

        linesOut.emplace_back(string_format(
                "goal distance:     [% 3.2f,% 3.2f] m",
                obs->ele[6], obs->ele[7]));

        auto omColl = observationModel->findOffsets<OMCollisionCost>();
        if (omColl) {
            linesOut.emplace_back(string_format(
                    "collision cost:        % 3.2f",
                    obs->ele[omColl.pos]));
        }

        auto omCollPred = observationModel->findOffsets<OMCollisionCostPrediction>();
        if (omCollPred) {
            linesOut.emplace_back(string_format(
                    "collision cost (pred): % 3.2f",
                    obs->ele[omCollPred.pos]));
        }
    }

};

// Register
static ExperimentConfigRegistration<ECTargetTracking> RegTargetTracking("TargetTracking");

}
