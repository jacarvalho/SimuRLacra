#ifndef _OMBODYSTATELINEAR_H_
#define _OMBODYSTATELINEAR_H_

#include "OMTask.h"
#include "OMTaskPositions.h"

namespace Rcs
{

/**
 * Observation model of linear body state.
 * Observes the position of the body along all three axis as well as the linear velocity.
 */
class OMBodyStateLinear : public OMTask
{
public:
    /**
     * Constructor
     *
     * @param graph        World to observe
     * @param effectorName Name of effector body, a.k.a. the body controlled by the task.
     * @param refBodyName  Name of reference body, a.k.a. the body the task coordinates should be relative to.
     *                     Set to NULL to use the world origin.
     * @param refFrameName Name of the reference frame body. The task coordinates will be expressed in this body's
     *                     frame if set. If this is NULL, refBodyName will be used.
     */
    OMBodyStateLinear(RcsGraph* graph, const char* effectorName, const char* refBodyName = NULL,
                      const char* refFrameName = NULL);
};

/**
 * Observation model of linear body state.
 * Observes the position of the body along all three axis, but not the linear velocity.
 */
class OMBodyStateLinearPositions : public OMTaskPositions
{
public:
    /**
     * Constructor
     *
     * @param graph        World to observe
     * @param effectorName Name of effector body, a.k.a. the body controlled by the task.
     * @param refBodyName  Name of reference body, a.k.a. the body the task coordinates should be relative to.
     *                     Set to NULL to use the world origin.
     * @param refFrameName Name of the reference frame body. The task coordinates will be expressed in this body's
     *                     frame if set. If this is NULL, refBodyName will be used.
     */
    OMBodyStateLinearPositions(RcsGraph* graph, const char* effectorName, const char* refBodyName = NULL,
                               const char* refFrameName = NULL);
};


} /* namespace Rcs */

#endif /* _OMBODYSTATELINEAR_H_ */
