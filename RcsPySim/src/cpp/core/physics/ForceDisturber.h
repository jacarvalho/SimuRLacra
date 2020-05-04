#ifndef _FORCEDISTURBER_H_
#define _FORCEDISTURBER_H_

#include <PhysicsBase.h>
#include <Rcs_graph.h>

namespace Rcs
{

class GraphNode;

/**
 * Applies a disturbing force to a specific body.
 */
class ForceDisturber
{
private:
    RcsBody* body;
    RcsBody* refFrame;

    //! Last applied force, in body coords, for GUI
    double lastForce[3];

public:
    ForceDisturber(RcsBody* body, RcsBody* refFrame=NULL);
    virtual ~ForceDisturber();

    void apply(Rcs::PhysicsBase* sim, double force[3]);

    void addToViewer(GraphNode* graphNode);

    const double *getLastForce() const;
};

} /* namespace Rcs */

#endif /* _FORCEDISTURBER_H_ */
