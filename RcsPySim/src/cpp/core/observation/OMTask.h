#ifndef _OMTASK_H_
#define _OMTASK_H_

#include "ObservationModel.h"

#include <Task.h>

namespace Rcs
{

/**
 * ObservationModel wrapping a Rcs Task.
 *
 * Note: By default, the state min/max is taken from the task, and the maximum velocity is set to infinity.
 * Use the various setters to change these limits. All limit setters return this for chanining.
 */
class OMTask: public ObservationModel
{
public:
    /**
     * Wrap the given task. Takes ownership of the task object.
     */
    OMTask(Task* task);
    virtual ~OMTask();

    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(OMTask)

    virtual unsigned int getStateDim() const;

    virtual void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const;

    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;

    virtual std::vector<std::string> getStateNames() const;

    /**
     * Set the lower state limit, broadcasting one value to all elements.
     */
    OMTask* setMinState(double minState);
    /**
     * Set the lower state limit. The number of elements must match the state dimension.
     */
    OMTask* setMinState(std::vector<double> minState);

    /**
     * Set the upper state limit, broadcasting one value to all elements.
     */
    OMTask* setMaxState(double maxState);
    /**
     * Set the upper state limit. The number of elements must match the state dimension.
     */
    OMTask* setMaxState(std::vector<double> maxState);

    /**
     * Set the velocity limit, broadcasting one value to all elements.
     */
    OMTask* setMaxVelocity(double maxVelocity);
    /**
     * Set the velocity limit. The number of elements must match the state dimension.
     */
    OMTask* setMaxVelocity(std::vector<double> maxVelocity);

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

    //! Settable maximum velocity (min/max state is stored in task parameter)
    std::vector<double> maxVelocity;
};

} /* namespace Rcs */

#endif /* _OMTASK_H_ */
