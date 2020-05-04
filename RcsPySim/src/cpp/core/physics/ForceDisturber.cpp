#include "ForceDisturber.h"

#include <Rcs_typedef.h>
#include <Rcs_Vec3d.h>

namespace Rcs
{

ForceDisturber::ForceDisturber(RcsBody* body, RcsBody* refFrame) : body(body), refFrame(refFrame)
{
    Vec3d_setZero(lastForce);
}

ForceDisturber::~ForceDisturber()
{
    // Nothing here to destroy
}

void ForceDisturber::apply(Rcs::PhysicsBase* sim, double force[3])
{
    // this is somewhat a bug in Rcs: The body map uses the bodies from the simulator's internal graph.
    // so, we need to convert this
    RcsBody* simBody = RcsGraph_getBodyByName(sim->getGraph(), body->name);

    // Transform force if needed
    double forceLocal[3];
    if (refFrame != nullptr)
    {
        Vec3d_rotate(forceLocal, refFrame->A_BI->rot, force);
        Vec3d_transRotateSelf(forceLocal, body->A_BI->rot);
    }
    else
    {
        Vec3d_copy(forceLocal, force);
    }

    // Store for UI
    Vec3d_copy(lastForce, force);

    // Apply the force in the physics simulation
    sim->setForce(simBody, force, NULL);
}

} /* namespace Rcs */

#ifdef GRAPHICS_AVAILABLE

#include <RcsViewer.h>
#include <GraphNode.h>

namespace Rcs
{

void ForceDisturber::addToViewer(GraphNode* graphNode)
{
    // Obtain graph node (assuming there's only one)
    BodyNode* bn = graphNode->getBodyNode(body);
}

const double *ForceDisturber::getLastForce() const
{
  return lastForce;
}

} /* namespace Rcs */

#else


void Rcs::ForceDisturber::addToViewer(GraphNode* graphNode)
{
    // nop
}

#endif
