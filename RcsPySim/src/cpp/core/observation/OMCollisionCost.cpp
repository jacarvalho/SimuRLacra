#include "OMCollisionCost.h"

#include <Rcs_collisionModel.h>
#include <Rcs_typedef.h>
#include <Rcs_VecNd.h>
#include <Rcs_macros.h>
#include <Rcs_resourcePath.h>
#include <Rcs_parser.h>

#include <sstream>
#include <stdexcept>

namespace Rcs
{

OMCollisionCost::OMCollisionCost(RcsCollisionMdl* collisionMdl, double maxCollCost) :
    collisionMdl(collisionMdl), maxCollCost(maxCollCost)
{
    // Debug
    REXEC(4)
    {
        RcsPair_printCollisionModel(stderr, collisionMdl->pair);
    }
}


OMCollisionCost::~OMCollisionCost()
{
    // Do not destroy the collision model since it is not owned
}

unsigned int OMCollisionCost::getStateDim() const
{
    return 1;
}

unsigned int OMCollisionCost::getVelocityDim() const
{
    return 0;
}

void OMCollisionCost::computeObservation(double* state, double* velocity, const MatNd* currentAction, double dt) const
//void Rcs::OMCollisionCost::computeState(double* state, const MatNd *currentAction, double dt) const
{
    // The state is the predicted collision cost
    RcsCollisionModel_compute(collisionMdl);
    state[0] = RcsCollisionMdl_cost(collisionMdl);
}


void OMCollisionCost::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    VecNd_setZero(minState, getStateDim()); // minimum cost is 0
    VecNd_setElementsTo(maxState, maxCollCost, getStateDim());  // maximum cost (theoretically infinite)
}

std::vector<std::string> OMCollisionCost::getStateNames() const
{
    return {"CollCost"};
}

} /* namespace Rcs */
