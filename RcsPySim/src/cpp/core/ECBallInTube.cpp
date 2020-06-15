#include "ExperimentConfig.h"
#include "action/ActionModelIK.h"
#include "action/AMTaskActivation.h"
#include "initState/ISSBallInTube.h"
#include "observation/OMCombined.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMBodyStateAngular.h"
#include "observation/OMGoalDistance.h"
#include "observation/OMForceTorque.h"
#include "observation/OMPartial.h"
#include "observation/OMCollisionCost.h"
#include "observation/OMCollisionCostPrediction.h"
#include "observation/OMManipulabilityIndex.h"
#include "observation/OMDynamicalSystemDiscrepancy.h"
#include "observation/OMTaskSpaceDiscrepancy.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"
#include "physics/PPDSphereRadius.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>
#include <Rcs_Vec3d.h>
#include <TaskPosition3D.h>
#include <TaskVelocity1D.h>
#include <TaskDistance.h>
#include <TaskOmega1D.h>
#include <TaskEuler3D.h>
#include <TaskFactory.h>

#ifdef GRAPHICS_AVAILABLE
#include <RcsViewer.h>
#endif

#include <memory>
#include <iomanip>

namespace Rcs
{
class ECBallInTube : public ExperimentConfig
{
protected:
    virtual ActionModel* createActionModel()
    {
        // Setup inner action model
        RcsBody* leftEffector = RcsGraph_getBodyByName(graph, "Effector_L");
        RCHECK(leftEffector);
        RcsBody* rightEffector = RcsGraph_getBodyByName(graph, "Effector_R");
        RCHECK(rightEffector);
        
        // Get reference frames for the position and orientation tasks
        std::string refFrameType = "world";
        properties->getProperty(refFrameType, "refFrame");
        RcsBody* refBody = nullptr;
        RcsBody* refFrame = nullptr;
        if (refFrameType == "world")
        {
            // Keep nullptr
        }
        else if (refFrameType == "table")
        {
            RcsBody* table = RcsGraph_getBodyByName(graph, "Table");
            RCHECK(table);
            refBody = table;
            refFrame = table;
        }
        else if (refFrameType == "slider")
        {
            RcsBody* slider = RcsGraph_getBodyByName(graph, "Slider");
            RCHECK(slider);
            refBody = slider;
            refFrame = slider;
        }
        else
        {
            std::ostringstream os;
            os << "Unsupported reference frame type: " << refFrame;
            throw std::invalid_argument(os.str());
        }
        
        // Initialize action model and tasks
        std::unique_ptr<AMIKGeneric> innerAM(new AMIKGeneric(graph));
        std::vector<std::unique_ptr<DynamicalSystem>> tasks;

        // Control effector positions and orientation
        if (properties->getPropertyBool("positionTasks", false))
        {
            RcsBody* slider = RcsGraph_getBodyByName(graph, "Slider");
            RCHECK(slider);
            // Left
            innerAM->addTask(new TaskPosition3D(graph, leftEffector, slider, slider));
            innerAM->addTask(new TaskEuler3D(graph, leftEffector, slider, slider));
            // Right
            innerAM->addTask(new TaskPosition3D(graph, rightEffector, slider, slider));
            innerAM->addTask(new TaskEuler3D(graph, rightEffector, slider, slider));

            // Obtain task data (depends on the order of the MPs coming from Pyrado)
            // Left
            unsigned int i = 0;
            std::vector<unsigned int> taskDimsLeft{
                3, 3, 3, 3
            };
            std::vector<unsigned int> offsetsLeft{
                0, 0, 3, 3
            };
            auto& tsLeft = properties->getChildList("tasksLeft");
            for (auto tsk : tsLeft)
            {
                DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsLeft[i]);
                tasks.emplace_back(new DSSlice(ds, offsetsLeft[i], taskDimsLeft[i]));
                i++;
            }
            // Right
            std::vector<unsigned int> taskDimsRight{
                3, 3, 3, 3
            };
            unsigned int oL = offsetsLeft.back() + taskDimsLeft.back();
            std::vector<unsigned int> offsetsRight{
                oL, oL, oL + 3, oL + 3
            };
            i = 0;
            auto& tsRight = properties->getChildList("tasksRight");
            for (auto tsk : tsRight)
            {
                DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsRight[i]);
                tasks.emplace_back(new DSSlice(ds, offsetsRight[i], taskDimsRight[i]));
                i++;
            }
        }

        // Control effector velocity and orientation
        else
        {
            // Left
            innerAM->addTask(new TaskVelocity1D("Xd", graph, leftEffector, refBody, refFrame));
            innerAM->addTask(new TaskVelocity1D("Yd", graph, leftEffector, refBody, refFrame));
            innerAM->addTask(new TaskVelocity1D("Zd", graph, leftEffector, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Ad", graph, leftEffector, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Bd", graph, leftEffector, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Cd", graph, leftEffector, refBody, refFrame));
            // Right
            innerAM->addTask(new TaskVelocity1D("Xd", graph, rightEffector, refBody, refFrame));
            innerAM->addTask(new TaskVelocity1D("Yd", graph, rightEffector, refBody, refFrame));
            innerAM->addTask(new TaskVelocity1D("Zd", graph, rightEffector, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Ad", graph, rightEffector, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Bd", graph, rightEffector, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Cd", graph, rightEffector, refBody, refFrame));

            // Obtain task data (depends on the order of the MPs coming from Pyrado)
            // Left
            unsigned int i = 0;
            std::vector<unsigned int> taskDimsLeft{
                1, 1, 1, 1, 1, 1
            };
            std::vector<unsigned int> offsetsLeft{
                0, 1, 2, 3, 4, 5
            };
            auto& tsLeft = properties->getChildList("tasksLeft");
            for (auto tsk : tsLeft)
            {
                DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsLeft[i]);
                tasks.emplace_back(new DSSlice(ds, offsetsLeft[i], taskDimsLeft[i]));
                i++;
            }
            // Right
            std::vector<unsigned int> taskDimsRight{
                1, 1, 1, 1, 1, 1
            };
            unsigned int oL = offsetsLeft.back() + taskDimsLeft.back();
            std::vector<unsigned int> offsetsRight{
                oL, oL + 1, oL + 2, oL + 3, oL + 4, oL + 5
            };
            i = 0;
            auto& tsRight = properties->getChildList("tasksRight");
            for (auto tsk : tsRight)
            {
                DynamicalSystem* ds = DynamicalSystem::create(tsk, taskDimsRight[i]);
                tasks.emplace_back(new DSSlice(ds, offsetsRight[i], taskDimsRight[i]));
                i++;
            }
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
        
        return new AMTaskActivation(innerAM.release(), taskRel, tcm);
    }

    virtual ObservationModel* createObservationModel()
    {
        // Observe effector positions (and velocities)
        std::unique_ptr<OMCombined> fullState(new OMCombined());
        
        auto omLeftLin = new OMBodyStateLinear(graph, "Effector_L"); // in world coordinates
        omLeftLin->setMinState({0., -1.6, 0.75});  // [m]
        omLeftLin->setMaxState({1.6, 1.6, 1.5});  // [m]
        omLeftLin->setMaxVelocity(3.); // [m/s]
        fullState->addPart(omLeftLin);

        auto omRightLin = new OMBodyStateLinear(graph, "Effector_R"); // in world coordinates
        omRightLin->setMinState({0., -1.6, 0.75});  // [m]
        omRightLin->setMaxState({1.6, 1.6, 1.5});  // [m]
        omRightLin->setMaxVelocity(3.); // [m/s]
        fullState->addPart(omRightLin);
        
        // Observe box positions (and velocities)
        auto omBallLin = new OMBodyStateLinear(graph, "Ball", "Table", "Table");  // in relative coordinates
        omBallLin->setMinState({-0.6, -0.8, -0.1});  // [m]
        omBallLin->setMaxState({0.6, 0.8, 0.1});  // [m]
//        auto omBallLin = new OMBodyStateLinear(graph, "Ball"); // in world coordinates
//        omBallLin->setMinState({0.9, -0.8, 0.66});  // [m]
//        omBallLin->setMaxState({2.1, 0.8, 1.26});  // [m]
        omBallLin->setMaxVelocity(5.); // [m/s]
        fullState->addPart(omBallLin);

        // Add goal distances
        if (properties->getPropertyBool("observeDSGoalDistance", false))
        {
            auto amAct = actionModel->unwrap<AMTaskActivation>();
            RCHECK(amAct);
            fullState->addPart(new OMGoalDistance(amAct));
        }
        
        // Add force/torque measurements
        if (properties->getPropertyBool("observeForceTorque", true))
        {
            RcsSensor* ftsL = RcsGraph_getSensorByName(graph, "WristLoadCellLBR_L");
            if (ftsL)
            {
                auto omForceTorque = new OMForceTorque(graph, ftsL->name, 1200);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {true, true, true, false, false, false}));
            }
            RcsSensor* ftsR = RcsGraph_getSensorByName(graph, "WristLoadCellLBR_R");
            if (ftsR)
            {
                auto omForceTorque = new OMForceTorque(graph, ftsR->name, 1200);
                fullState->addPart(OMPartial::fromMask(omForceTorque, {true, true, true, false, false, false}));
            }
        }
        
        // Add current collision cost
        if (properties->getPropertyBool("observeCollisionCost", true) & (collisionMdl != nullptr))
        {
            // Add the collision cost observation model
            auto omColl = new OMCollisionCost(collisionMdl);
            fullState->addPart(omColl);
        }
        
        // Add predicted collision cost
        if (properties->getPropertyBool("observePredictedCollisionCost", false) & (collisionMdl != nullptr))
        {
            // Get the horizon from config
            int horizon = 20;
            properties->getChild("collisionConfig")->getProperty(horizon, "predCollHorizon");
            // Add the collision cost observation model
            auto omCollPred = new OMCollisionCostPrediction(graph, collisionMdl, actionModel, horizon);
            fullState->addPart(omCollPred);
        }
        
        // Add manipulability index
        auto ikModel = actionModel->unwrap<ActionModelIK>();
        if (properties->getPropertyBool("observeManipulabilityIndex", false) && ikModel)
        {
            bool ocm = properties->getPropertyBool("observeCurrentManipulability", true);
            fullState->addPart(new OMManipulabilityIndex(ikModel, ocm));
        }
        
        // Add the dynamical system discrepancy observation model
        if (properties->getPropertyBool("observeDynamicalSystemDiscrepancy", false) & (collisionMdl != nullptr))
        {
            auto castedAM = dynamic_cast<AMTaskActivation*>(actionModel);
            if (castedAM)
            {
                auto omDSDescr = new OMDynamicalSystemDiscrepancy(castedAM);
                fullState->addPart(omDSDescr);
            }
            else
            {
                throw std::invalid_argument("The action model needs to be of type AMTaskActivation!");
            }
        }
        
        // Add the task space discrepancy observation model
        if (properties->getPropertyBool("observeTaskSpaceDiscrepancy", true))
        {
            auto wamIK = actionModel->unwrap<ActionModelIK>();
            if (wamIK)
            {
                auto omTSDescrL = new OMTaskSpaceDiscrepancy("Effector_L", graph, wamIK->getController()->getGraph());
                fullState->addPart(omTSDescrL);
                auto omTSDescrR = new OMTaskSpaceDiscrepancy("Effector_R", graph, wamIK->getController()->getGraph());
                fullState->addPart(omTSDescrR);
            }
            else
            {
                throw std::invalid_argument("The action model needs to be of type ActionModelIK!");
            }
        }
        
        return fullState.release();
    }
    
    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Ball", new PPDSphereRadius());
        manager->addParam("Ball", new PPDMassProperties());
        manager->addParam("Ball", new PPDMaterialProperties());
        manager->addParam("Slider", new PPDMassProperties());
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSBallInTube(graph);
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* box = RcsGraph_getBodyByName(graph, "Ball");
        RCHECK(box);
        return new ForceDisturber(box, box);
    }
    
    virtual void initViewer(Rcs::Viewer* viewer)
    {
#ifdef GRAPHICS_AVAILABLE
        // Set camera above plate
        RcsBody* table = RcsGraph_getBodyByName(graph, "Table");
        RCHECK(table);
        std::string cameraView = "egoView";
        properties->getProperty(cameraView, "egoView");
        
        // The camera center is 10cm above the the plate's center
        double cameraCenter[3];
        Vec3d_copy(cameraCenter, table->A_BI->org);
        cameraCenter[0] -= 0.2;

        // The camera location - not specified yet
        double cameraLocation[3];
        Vec3d_setZero(cameraLocation);
        
        // Camera up vector defaults to z
        double cameraUp[3];
        Vec3d_setUnitVector(cameraUp, 2);
        
        if (cameraView == "egoView")
        {
            RcsBody* railBot = RcsGraph_getBodyByName(graph, "RailBot");
            RCHECK(railBot);

            // Rotate to world frame
            Vec3d_transformSelf(cameraLocation, table->A_BI);

            // Move the camera approx where the Kinect would be
            cameraLocation[0] = railBot->A_BI->org[0] - 0.5;
            cameraLocation[1] = railBot->A_BI->org[1];
            cameraLocation[2] = railBot->A_BI->org[2] + 1.5;
        }
        else
        {
            RMSG("Unsupported camera view: %s", cameraView.c_str());
            return;
        }
        
        // Apply the camera position
        viewer->setCameraHomePosition(osg::Vec3d(cameraLocation[0], cameraLocation[1], cameraLocation[2]),
                                      osg::Vec3d(cameraCenter[0], cameraCenter[1], cameraCenter[2]),
                                      osg::Vec3d(cameraUp[0], cameraUp[1], cameraUp[2]));
#endif
    }
    
    void
    getHUDText(
        std::vector<std::string>& linesOut,
        double currentTime, const MatNd* obs,
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
            string_format("physics engine: %s                     sim time:        %2.3f s", simName, currentTime));
        
        unsigned int numPosCtrlJoints = 0;
        unsigned int numTrqCtrlJoints = 0;
        // Iterate over unconstrained joints
        RCSGRAPH_TRAVERSE_JOINTS(graph)
        {
            if (JNT->jacobiIndex != -1)
            {
                if (JNT->ctrlType == RCSJOINT_CTRL_TYPE::RCSJOINT_CTRL_POSITION)
                {
                    numPosCtrlJoints++;
                }
                else if (JNT->ctrlType == RCSJOINT_CTRL_TYPE::RCSJOINT_CTRL_TORQUE)
                {
                    numTrqCtrlJoints++;
                }
            }
        }
        linesOut.emplace_back(string_format("num joints:    %d total, %d pos ctrl, %d trq ctrl",
                             graph->nJ, numPosCtrlJoints, numTrqCtrlJoints));

        unsigned int sd = observationModel->getStateDim();

        auto omLeftLin = observationModel->findOffsets<OMBodyStateLinear>(); // there are two, we find the first
        if (omLeftLin)
        {
            linesOut.emplace_back(
                string_format("left hand pg:  [% 1.3f,% 1.3f,% 1.3f] m   [% 1.3f,% 1.3f,% 1.3f] m/s",
                    obs->ele[omLeftLin.pos], obs->ele[omLeftLin.pos + 1], obs->ele[omLeftLin.pos + 2],
                    obs->ele[sd + omLeftLin.vel], obs->ele[sd + omLeftLin.vel + 1], obs->ele[sd + omLeftLin.vel + 2]));
            linesOut.emplace_back(
                string_format("right hand pg: [% 1.3f,% 1.3f,% 1.3f] m   [% 1.3f,% 1.3f,% 1.3f] m/s",
                    obs->ele[omLeftLin.pos + 3], obs->ele[omLeftLin.pos + 4], obs->ele[omLeftLin.pos + 5],
                    obs->ele[sd + omLeftLin.vel + 3], obs->ele[sd + omLeftLin.vel + 4],
                    obs->ele[sd + omLeftLin.vel + 5]));
        }

        else if (omLeftLin)
        {
            linesOut.emplace_back(
                string_format("box absolute:  [% 1.3f,% 1.3f,% 1.3f] m",
                    obs->ele[omLeftLin.pos + 6], obs->ele[omLeftLin.pos + 7], obs->ele[omLeftLin.pos + 8]));
        }
        
        auto omFTS = observationModel->findOffsets<OMForceTorque>();
        if (omFTS)
        {
            linesOut.emplace_back(
                string_format("forces left:   [% 3.1f, % 3.1f, % 3.1f] N     right: [% 3.1f, % 3.1f, % 3.1f] N",
                    obs->ele[omFTS.pos], obs->ele[omFTS.pos + 1], obs->ele[omFTS.pos + 2],
                    obs->ele[omFTS.pos + 3], obs->ele[omFTS.pos + 4], obs->ele[omFTS.pos + 5]));
        }
        
        auto omColl = observationModel->findOffsets<OMCollisionCost>();
        auto omCollPred = observationModel->findOffsets<OMCollisionCostPrediction>();
        if (omColl && omCollPred)
        {
            linesOut.emplace_back(
                string_format("coll cost:       %3.2f                    pred coll cost: %3.2f",
                    obs->ele[omColl.pos], obs->ele[omCollPred.pos]));
            
        }
        else if (omColl)
        {
            linesOut.emplace_back(string_format("coll cost:       %3.2f", obs->ele[omColl.pos]));
        }
        else if (omCollPred)
        {
            linesOut.emplace_back(string_format("pred coll cost:   %3.2f", obs->ele[omCollPred.pos]));
        }
        
        auto omMI = observationModel->findOffsets<OMManipulabilityIndex>();
        if (omMI)
        {
            linesOut.emplace_back(string_format("manip idx:       %1.3f", obs->ele[omMI.pos]));
        }

        auto omGD = observationModel->findOffsets<OMGoalDistance>();
        if (omGD)
        {
            if (properties->getPropertyBool("positionTasks", false)) // TODO
            {
                linesOut.emplace_back(
                    string_format("goal distance: [% 1.2f,% 1.2f,% 1.2f,% 1.2f,% 1.2f,\n"
                                          "               % 1.2f,% 1.2f,% 1.2f,% 1.2f,% 1.2f,% 1.2f]",
                        obs->ele[omGD.pos], obs->ele[omGD.pos + 1], obs->ele[omGD.pos + 2],
                        obs->ele[omGD.pos + 3], obs->ele[omGD.pos + 4],
                        obs->ele[omGD.pos + 5],  obs->ele[omGD.pos + 6],  obs->ele[omGD.pos + 7],
                        obs->ele[omGD.pos + 8],  obs->ele[omGD.pos + 9],  obs->ele[omGD.pos + 10]));
            }
        }

        auto omTSD = observationModel->findOffsets<OMTaskSpaceDiscrepancy>();
        if (omTSD)
        {
            linesOut.emplace_back(
                string_format("ts delta:      [% 1.3f,% 1.3f,% 1.3f] m",
                    obs->ele[omTSD.pos], obs->ele[omTSD.pos + 1], obs->ele[omTSD.pos + 2]));
        }
        
        std::stringstream ss;
        ss << "actions:       [";
        for (unsigned int i = 0; i < currentAction->m - 1; i++)
        {
            ss << std::fixed << std::setprecision(2) << MatNd_get(currentAction, i, 0) << ", ";
            if (i == 6)
            {
                ss << "\n                ";
            }
        }
        ss << std::fixed << std::setprecision(2) << MatNd_get(currentAction, currentAction->m - 1, 0) << "]";
        linesOut.emplace_back(string_format(ss.str()));
    
        if (physicsManager != nullptr)
        {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* ball_bpi = physicsManager->getBodyInfo("Ball");
            double* com = ball_bpi->body->Inertia->org;
            double ball_radius = ball_bpi->body->shape[0]->extents[0];
            double slip = 0;
            ball_bpi->material.getDouble("slip", slip);

            linesOut.emplace_back(string_format(
                "ball mass:      %2.2f kg           ball radius:             %2.3f cm",
                ball_bpi->body->m, ball_radius * 100));

            linesOut.emplace_back(string_format(
                "ball friction:  %1.2f    ball rolling friction:             %1.3f",
                ball_bpi->material.getFrictionCoefficient(),
                ball_bpi->material.getRollingFrictionCoefficient() / ball_radius));

            linesOut.emplace_back(string_format(
                "ball slip:      %3.1f rad/(Ns)       CoM offset:[% 2.1f, % 2.1f, % 2.1f] mm",
                slip, com[0] * 1000, com[1] * 1000, com[2] * 1000));
        }
    }
    
};

// Register
static ExperimentConfigRegistration<ECBallInTube> RegBallInTube("BallInTube");
    
}
