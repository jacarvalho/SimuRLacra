#include "AMPlatePos5D.h"

#include <TaskPosition3D.h>
#include <TaskEuler3D.h>

#include <Rcs_macros.h>
#include <Rcs_VecNd.h>
#include <Rcs_HTr.h>
#include <Rcs_typedef.h>


namespace Rcs
{
const unsigned int PLATE_TASK_DIM = 6;

AMPlatePos5D::AMPlatePos5D(RcsGraph* graph) : ActionModelIK(graph)
{
    // lookup plate body on desired state graph
    RcsBody* plate = RcsGraph_getBodyByName(desiredGraph, "Plate");
    RCHECK(plate);

    // the plate position is computed relative to a marker frame, which is set to
    // the plate's initial position on startup.
    RcsBody* plateOrigMarker = RcsGraph_getBodyByName(desiredGraph, "PlateOriginMarker");
    RCHECK_MSG(plateOrigMarker, "PlateOriginMarker is missing, please update your xml.");
    // set origin marker position
    HTr_copyOrRecreate(&plateOrigMarker->A_BP, plate->A_BI);

    // create dynamicalSystems relative to origin marker
    addTask(new TaskPosition3D(desiredGraph, plate, plateOrigMarker, NULL));
    addTask(new TaskEuler3D(desiredGraph, plate, plateOrigMarker, NULL));
    RCHECK_EQ(PLATE_TASK_DIM, getController()->getTaskDim());
}

AMPlatePos5D::~AMPlatePos5D()
{
}

unsigned int AMPlatePos5D::getDim() const
{
    return 5;
}

void AMPlatePos5D::getMinMax(double* min, double* max) const
{
    // separate maximal divergence for XY movement, Z movement and angular movement.

    double maxXY = 0.5;
    min[0] = -maxXY;
    min[1] = -maxXY;
    max[0] = maxXY;
    max[1] = maxXY;

    double maxZ = 0.5;
    min[2] = -maxZ;
    max[2] = maxZ;

    double maxAngle = 45 * M_PI / 180;
    min[4] = -maxAngle;
    min[5] = -maxAngle;
    max[4] = maxAngle;
    max[5] = maxAngle;
}

std::vector<std::string> AMPlatePos5D::getNames() const
{
    return { "x", "y", "z", "a", "b" };
}

void AMPlatePos5D::computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt)
{
    // expand action into state matrix by adding the desired Z rotation which is always 0
    MatNd* x_des;
    MatNd_fromStack(x_des, PLATE_TASK_DIM, 1);
    VecNd_copy(x_des->ele, action->ele, 5);

    // delegate to IK
    computeIK(q_des, q_dot_des, T_des, x_des, dt);
}

void AMPlatePos5D::getStableAction(MatNd* action) const
{
    // stable action == keeps current state
    MatNd* x_curr;
    MatNd_fromStack(x_curr, PLATE_TASK_DIM, 1);
    // Compute current task state
    getController()->computeX(x_curr);

    // copy parts of state relevant for the action
    VecNd_copy(action->ele, x_curr->ele, 5);
}

ActionModel *AMPlatePos5D::clone(RcsGraph *newGraph) const
{
    auto res = new AMPlatePos5D(newGraph);
    res->setupCollisionModel(collisionMdl);
    return res;
}

} /* namespace Rcs */
