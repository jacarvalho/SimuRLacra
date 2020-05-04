#ifndef RCSPYSIM_POLICYCOMPONENT_H
#define RCSPYSIM_POLICYCOMPONENT_H


#include <config/PropertySource.h>
#include <ExperimentConfig.h>
#include <control/ControlPolicy.h>

#include <ComponentBase.h>
#include <Rcs_MatNd.h>

namespace Rcs {

class DynamicalSystem;

/**
 * Wraps RcsPySim Experiment and ControlPolicy for use in ECS.
 */
class PolicyComponent : public ComponentBase
{
public:
    PolicyComponent(EntityBase* entity, PropertySource* settings, bool computeJointVelocities=false);

    virtual ~PolicyComponent();

    // not copy- or movable
    RCSPYSIM_NOCOPY_NOMOVE(PolicyComponent)

    ExperimentConfig* getExperiment() const;

    ControlPolicy* getPolicy() const;

    const MatNd* getObservation() const;

    const MatNd* getAction() const;

    const MatNd* getJointCommandPtr() const;

    RcsGraph* getDesiredGraph() const;

    // get a text describing the current state of the command fsm
    std::string getStateText() const;

    void setJointLimitCheck(bool jointLimitCheck);
    void setCollisionCheck(bool collisionCheck);

private:
    // event handlers
    void subscribe();

//    void onStart();
//    void onStop();
    void onUpdatePolicy(const RcsGraph* state);
    void onInitFromState(const RcsGraph* target);
    void onEmergencyStop();
    void onEmergencyRecover();
    void onRender();
    void onPrint();
    void onPolicyStart();
    void onPolicyPause();
    void onPolicyReset();
    void onGoHome();


    // experiment to use
    ExperimentConfig* experiment;
    // policy to use
    ControlPolicy* policy;

    // the robot doesn't provide joint velocities. to work around that, compute them using finite differences
    bool computeJointVelocities;
    
    // true to check joint limits
    bool jointLimitCheck;
    // true to check for collisions
    bool collisionCheck;

    // collision model for collision check
    RcsCollisionMdl* collisionMdl;

    // true if the policy should be active
    bool policyActive;
    // true while in emergency stop
    bool eStop;
    // true in the first onUpdatePolicy call after onEmergencyRecover
    bool eRec;
    // allows to catch the first render call
    bool renderingInitialized;


    // Temporary matrices
    MatNd* observation;
    MatNd* action;

    // graph containing the desired states
    RcsGraph* desiredGraph;

    // go home policy
    bool goHome;
    DynamicalSystem* goHomeDS;
    ActionModel* goHomeAM;

    // dummy, might be filled but is not used
    MatNd* T_ctrl;

};

}

#endif //RCSPYSIM_POLICYCOMPONENT_H
