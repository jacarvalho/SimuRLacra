#include "ExperimentConfig.h"
#include "action/ActionModelIK.h"
#include "action/AMTaskActivation.h"
#include "initState/ISSBoxLifting.h"
#include "observation/OMCombined.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMBodyStateAngular.h"
#include "observation/OMDynamicalSystemGoalDistance.h"
#include "observation/OMForceTorque.h"
#include "observation/OMPartial.h"
#include "observation/OMCollisionCost.h"
#include "observation/OMCollisionCostPrediction.h"
#include "observation/OMManipulabilityIndex.h"
#include "observation/OMDynamicalSystemDiscrepancy.h"
#include "observation/OMTaskSpaceDiscrepancy.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDBoxExtents.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"

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
class ECBoxLifting : public ExperimentConfig
{
protected:
    virtual ActionModel* createActionModel()
    {
        // Setup inner action model
        RcsBody* leftGrasp = RcsGraph_getBodyByName(graph, "PowerGrasp_L");
        RCHECK(leftGrasp);
        RcsBody* rightGrasp = RcsGraph_getBodyByName(graph, "PowerGrasp_R");
        RCHECK(rightGrasp);
        
        // Get reference frames for the position and orientation tasks
        std::string refFrameType = "world";
        properties->getProperty(refFrameType, "refFrame");
        RcsBody* refBody = nullptr;
        RcsBody* refFrame = nullptr;
        if (refFrameType == "world")
        {
            // Keep nullptr
        }
        else if (refFrameType == "box")
        {
            RcsBody* box = RcsGraph_getBodyByName(graph, "Box");
            RCHECK(box);
            refBody = box;
            refFrame = box;
        }
        else if (refFrameType == "basket")
        {
            RcsBody* basket = RcsGraph_getBodyByName(graph, "Basket");
            RCHECK(basket);
            refBody = basket;
            refFrame = basket;
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
            RcsBody* basket = RcsGraph_getBodyByName(graph, "Basket");
            RCHECK(basket);
            // Left
            innerAM->addTask(new TaskPosition3D(graph, leftGrasp, basket, basket));
            innerAM->addTask(new TaskEuler3D(graph, leftGrasp, basket, basket));
            innerAM->addTask(TaskFactory::createTask(
                "<Task name=\"Hand L Joints\" effector=\"PowerGrasp_L\" controlVariable=\"Joints\" jnts=\"fing1-knuck1_L tip1-fing1_L fing2-knuck2_L tip2-fing2_L fing3-knuck3_L tip3-fing3_L knuck1-base_L\" tmc=\"0.1\" vmax=\"1000\" active=\"untrue\"/>", graph)
            );
            // Right
            innerAM->addTask(new TaskPosition3D(graph, rightGrasp, basket, basket));
            innerAM->addTask(new TaskEuler3D(graph, rightGrasp, basket, basket));
            innerAM->addTask(TaskFactory::createTask(
                "<Task name=\"Hand R Joints\" effector=\"PowerGrasp_R\" controlVariable=\"Joints\" jnts=\"fing1-knuck1_R tip1-fing1_R fing2-knuck2_R tip2-fing2_R fing3-knuck3_R tip3-fing3_R knuck1-base_R\" tmc=\"0.1\" vmax=\"1000\" active=\"untrue\"/>", graph)
            );
            innerAM->addTask(TaskFactory::createTask(
                "<Task name=\"Distance R\" controlVariable=\"Distance\"  effector=\"tip2_R\" refBdy=\"Box\" gainDX=\"1.\" active=\"true\"/>", graph)
            );

            // Obtain task data (depends on the order of the MPs coming from Pyrado)
            // Left
            unsigned int i = 0;
            std::vector<unsigned int> taskDimsLeft{
                3, 3, 3, 3, 7
            };
            std::vector<unsigned int> offsetsLeft{
                0, 0, 3, 3, 6
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
                3, 3, 3, 3, 7, 1
            };
            unsigned int oL = offsetsLeft.back() + taskDimsLeft.back();
            std::vector<unsigned int> offsetsRight{
                oL, oL, oL + 3, oL + 3, oL + 6, oL + 13,
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
            innerAM->addTask(new TaskVelocity1D("Xd", graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(new TaskVelocity1D("Yd", graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(new TaskVelocity1D("Zd", graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Ad", graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Bd", graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Cd", graph, leftGrasp, refBody, refFrame));
            innerAM->addTask(TaskFactory::createTask(
                "<Task name=\"Hand L Joints\" effector=\"PowerGrasp_L\" controlVariable=\"Joints\" jnts=\"fing1-knuck1_L tip1-fing1_L fing2-knuck2_L tip2-fing2_L fing3-knuck3_L tip3-fing3_L knuck1-base_L\" tmc=\"0.1\" vmax=\"1000\" active=\"untrue\"/>", graph)
                );
            // Right
            innerAM->addTask(new TaskVelocity1D("Xd", graph, rightGrasp, refBody, refFrame));
            innerAM->addTask(new TaskVelocity1D("Yd", graph, rightGrasp, refBody, refFrame));
            innerAM->addTask(new TaskVelocity1D("Zd", graph, rightGrasp, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Ad", graph, rightGrasp, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Bd", graph, rightGrasp, refBody, refFrame));
            innerAM->addTask(new TaskOmega1D("Cd", graph, rightGrasp, refBody, refFrame));
            //        innerAM->addTask(new TaskJoint(graph, RcsGraph_getJointByName(graph, "fing1-knuck1_R")));
            innerAM->addTask(TaskFactory::createTask(
                "<Task name=\"Hand R Joints\" effector=\"PowerGrasp_R\" controlVariable=\"Joints\" jnts=\"fing1-knuck1_R tip1-fing1_R fing2-knuck2_R tip2-fing2_R fing3-knuck3_R tip3-fing3_R knuck1-base_R\" tmc=\"0.1\" vmax=\"1000\" active=\"untrue\"/>", graph)
                );

            // Obtain task data (depends on the order of the MPs coming from Pyrado)
            // Left
            unsigned int i = 0;
            std::vector<unsigned int> taskDimsLeft{
                1, 1, 1, 1, 1, 1, 7
            };
            std::vector<unsigned int> offsetsLeft{
                0, 1, 2, 3, 4, 5, 6
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
                1, 1, 1, 1, 1, 1, 7
            };
            unsigned int oL = offsetsLeft.back() + taskDimsLeft.back();
            std::vector<unsigned int> offsetsRight{
                oL, oL + 1, oL + 2, oL + 3, oL + 4, oL + 5, oL + 6
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
            REXEC(4)
                {
                    std::cout << "IK considers the provided collision model" << std::endl;
                }
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
        
        auto omLeftLin = new OMBodyStateLinear(graph, "PowerGrasp_L"); // in world coordinates
        omLeftLin->setMinState({0., -1.6, 0.75});  // [m]
        omLeftLin->setMaxState({1.6, 1.6, 1.5});  // [m]
        omLeftLin->setMaxVelocity(3.); // [m/s]
        fullState->addPart(omLeftLin);

        auto omRightLin = new OMBodyStateLinear(graph, "PowerGrasp_R"); // in world coordinates
        omRightLin->setMinState({0., -1.6, 0.75});  // [m]
        omRightLin->setMaxState({1.6, 1.6, 1.5});  // [m]
        omRightLin->setMaxVelocity(3.); // [m/s]
        fullState->addPart(omRightLin);
        
        // Observe box positions (and velocities)
//        auto omBoxLin = new OMBodyStateLinear(graph, "Box", "Table", "Table");  // in relative coordinates
//        omBoxLin->setMinState({-0.6, -0.8, -0.1});  // [m]
//        omBoxLin->setMaxState({0.6, 0.8, 1.});  // [m]
        auto omBoxLin = new OMBodyStateLinear(graph, "Box"); // in world coordinates
        omBoxLin->setMinState({0.9, -0.8, 0.66});  // [m]
        omBoxLin->setMaxState({2.1, 0.8, 1.26});  // [m]
        omBoxLin->setMaxVelocity(5.); // [m/s]
        fullState->addPart(omBoxLin);
        
        // Observe box orientations (and velocities)
        auto omBoxAng = new OMBodyStateAngular(graph, "Box"); // in world coordinates
        omBoxAng->setMaxVelocity(RCS_DEG2RAD(720)); // [rad/s]
        fullState->addPart(omBoxAng);
        
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
                auto omTSDescrL = new OMTaskSpaceDiscrepancy("PowerGrasp_L", graph, wamIK->getController()->getGraph());
                fullState->addPart(omTSDescrL);
                auto omTSDescrR = new OMTaskSpaceDiscrepancy("PowerGrasp_R", graph, wamIK->getController()->getGraph());
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
        manager->addParam("Box", new PPDBoxExtents(0, true, true, false)); // the Box body has only 1 shape
        manager->addParam("Box", new PPDMassProperties());
        manager->addParam("Box", new PPDMaterialProperties());
        manager->addParam("Basket", new PPDMassProperties());
        manager->addParam("Basket", new PPDMaterialProperties());
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSBoxLifting(graph);
    }
    
    virtual ForceDisturber* createForceDisturber()
    {
        RcsBody* box = RcsGraph_getBodyByName(graph, "Box");
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
        
        auto omBoxAng = observationModel->findOffsets<OMBodyStateAngular>(); // assuming there is only the box one
        if (omBoxAng && omLeftLin)
        {
            linesOut.emplace_back(
                string_format("box absolute:  [% 1.3f,% 1.3f,% 1.3f] m   [% 3.0f,% 3.0f,% 3.0f] deg",
                    obs->ele[omLeftLin.pos + 6], obs->ele[omLeftLin.pos + 7], obs->ele[omLeftLin.pos + 8],
                    obs->ele[omBoxAng.pos]*180/M_PI, obs->ele[omBoxAng.pos + 1]*180/M_PI,
                    obs->ele[omBoxAng.pos + 2]*180/M_PI));
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
            if (properties->getPropertyBool("positionTasks", false))
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
        
        auto castedAM = dynamic_cast<AMTaskActivation*>(actionModel);
        if (castedAM)
        {
            std::stringstream ss;
            ss << "activations:   [";
            for (unsigned int i = 0; i < castedAM->getDim() - 1; i++)
            {
                ss << std::fixed << std::setprecision(2) << MatNd_get(castedAM->getActivation(), i, 0) << ", ";
                if (i == 6)
                {
                    ss << "\n                ";
                }
            }
            ss << std::fixed << std::setprecision(2) <<
             MatNd_get(castedAM->getActivation(), castedAM->getDim() - 1, 0) << "]";
            linesOut.emplace_back(string_format(ss.str()));
            
            linesOut.emplace_back(string_format("tcm:            %s", castedAM->getTaskCombinationMethodName()));
        }
    
        if (physicsManager != nullptr)
        {
            // Get the parameters that are not stored in the Rcs graph
            BodyParamInfo* box_bpi = physicsManager->getBodyInfo("Box");
            BodyParamInfo* basket_bpi = physicsManager->getBodyInfo("Basket");

            linesOut.emplace_back(
                string_format("box width:   %1.3f m           box length: %1.3f m",
                              box_bpi->body->shape[0]->extents[0], box_bpi->body->shape[0]->extents[1]));
            linesOut.emplace_back(
                string_format("box mass:    %1.2f kg      box frict coeff: %1.3f  ",
                              box_bpi->body->m, box_bpi->material.getFrictionCoefficient()));
            linesOut.emplace_back(
                string_format("basket mass: %1.2f kg   basket frict coeff: %1.3f  ",
                              basket_bpi->body->m, basket_bpi->material.getFrictionCoefficient()));
        }
    }
    
};

// Register
static ExperimentConfigRegistration<ECBoxLifting> RegBoxLifting("BoxLifting");
    
}
