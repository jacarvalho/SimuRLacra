#ifndef _OMTASKPOSITIONS_H_
#define _OMTASKPOSITIONS_H_

#include "ObservationModel.h"

#include <Task.h>

namespace Rcs
{

/**
 * ObservationModel wrapping a Rcs Task.
 *
 * Note: In contrast to OMTask, this ObservationModel only observes the task space positions called X.
 */
class OMTaskPositions: public ObservationModel
{
public:
    /**
     * Wrap the given task. Takes ownership of the task object.
     */
    OMTaskPositions(Task* task);
    virtual ~OMTaskPositions();

    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(OMTaskPositions)

    virtual unsigned int getStateDim() const;

    virtual unsigned int getVelocityDim() const;

    virtual void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const;

    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;

    virtual std::vector<std::string> getStateNames() const;

    /**
     * Set the lower state limit, broadcasting one value to all elements.
     */
    OMTaskPositions* setMinState(double minState);
    /**
     * Set the lower state limit. The number of elements must match the state dimension.
     */
    OMTaskPositions* setMinState(std::vector<double> minState);

    /**
     * Set the upper state limit, broadcasting one value to all elements.
     */
    OMTaskPositions* setMaxState(double maxState);
    /**
     * Set the upper state limit. The number of elements must match the state dimension.
     */
    OMTaskPositions* setMaxState(std::vector<double> maxState);

    /**
     * Return the wrapped Rcs Task.
     */
    Task* getTask() const
    {
        return task;
    }

protected:
    /**
     * Initialize the task's effector, refBody and refFrame values by looking up the named bodies from the graph.
     *
     * @param effectorName Name of effector body, a.k.a. the body controlled by the task.
     * @param refBodyName  Name of reference body, a.k.a. the body the task coordinates
     *                     should be relative to. Set to NULL to use the world origin.
     * @param refFrameName Name of the reference frame body. The task coordinates will
     *                     be expressed in this body's frame if set. If this is NULL,
     *                     refBodyName will be used.
     */
    void initTaskBodyNames(const char* effectorName, const char* refBodyName, const char* refFrameName);

private:
    //! Wrapped task object (owned!)
    Task* task;
};

} /* namespace Rcs */

#endif /* _OMTASKPOSITIONS_H_ */
