#ifndef _RCSPYBOT_H_
#define _RCSPYBOT_H_

#include "ExperimentConfig.h"
#include "DataLogger.h"

#include <MotionControlLayer.h>

#include <mutex>

namespace Rcs
{

class ActionModel;
class ObservationModel;
class ControlPolicy;

// TODO besserer Name
class RcsPyBot: public MotionControlLayer
{
public:
    /**
     * Create the bot from the given property source.
     * @param propertySource configuration
     */
    explicit RcsPyBot(PropertySource* propertySource);
    virtual ~RcsPyBot();

    // not copy- or movable
    RCSPYSIM_NOCOPY_NOMOVE(RcsPyBot)

    ExperimentConfig* getConfig() {
        return config;
    }

    /**
     * Replace the control policy.
     * This method may be called while the bot is running.
     * Setting NULL causes the bot to return to it's initial position.
     * Does NOT take ownership.
     */
    void setControlPolicy(ControlPolicy* controlPolicy);
    ControlPolicy* getControlPolicy() const
    {
        std::unique_lock<std::mutex> lock(controlPolicyMutex);
        return controlPolicy;
    }

    //! Data logger
    DataLogger logger;

    /**
     * Get storage matrix for current observation.
     *
     * WARNING: the contents may update asynchronously. The dimensions are constant.
     */
    MatNd *getObservation() const;

    /**
     * Get storage matrix for current action.
     *
     * WARNING: the contents may update asynchronously. The dimensions are constant.
     */
    MatNd *getAction() const;

protected:
    virtual void updateControl();

    //! Control policy mutex (mutable to allow using it from const functions)
    mutable std::mutex controlPolicyMutex;

    //! Experiment configuration
    ExperimentConfig* config;
    //! Control policy
    ControlPolicy* controlPolicy;

    // temporary matrices
    MatNd* q_ctrl;
    MatNd* qd_ctrl;
    MatNd* T_ctrl;

    MatNd* observation;
    MatNd* action;
};

} /* namespace Rcs */

#endif /* _RCSPYBOT_H_ */
