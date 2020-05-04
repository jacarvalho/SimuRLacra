#ifndef _ACTIONMODEL_H_
#define _ACTIONMODEL_H_

#include "../util/BoxSpaceProvider.h"

#include <Rcs_MatNd.h>
#include <Rcs_graph.h>
#include <Rcs_typedef.h>

namespace Rcs
{

/**
 * The ActionModel component encapsulates the translation from the generic action passed from Python to actual joint
 * space commands required by the physics simulation or the robot.
 * Inherits from BoxSpaceProvider to provide a space for valid action values.
 */
class ActionModel : public BoxSpaceProvider
{
public:
    /**
     * Constructor
     * @param graph graph being commanded
     */
    explicit ActionModel(RcsGraph* graph);

    virtual ~ActionModel();

    /**
     * Create a deep copy of this action model.
     * @return deep copy.
     */
    inline ActionModel* clone() const
    { return clone(graph); }
    /**
     * Create a deep copy of this action model. The graph the action model operates on is replaced with newGraph.
     * @param newGraph optionally, replace the graph used with newGraph.
     * @return deep copy with new graph.
     */
    virtual ActionModel* clone(RcsGraph* newGraph) const = 0;

    /**
     * Compute the joint commands from a specified action and the current state.
     * @param[out] q_des desired joint positions
     * @param[out] q_dot_des desired joint velocities
     * @param[out] T_des desired joint torques
     * @param[in]  action input action values
     * @param      dt difference in time since the last call.
     */
    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt) = 0;

    /**
     * Called at the start of a rollout to reset any state modified by computeCommand().
     * This allows to reuse the ActionModel for a new simulation rollout. The graph state will already be reset,
     * so it can be used safely.
     * It will be called before the first rollout too, so it can also be used to setup internals that depend on
     * operations in subclass constructors.
     */
    virtual void reset();

    /**
     * Obtain action values which would keep the system in the current state.
     * For action variables which are velocities or accelerations, this should be 0.
     * For action variables which are positions, this should be the current position.
     * @param[out] action matrix to write the values into
     */
    virtual void getStableAction(MatNd* action) const = 0;
    
    /**
     * Get the graph this action model operates on.
     */
    RcsGraph* getGraph()
    { return graph; }
    
    /*!
     * If this ActionModel is a wrapper for another action model, return the wrapped action model.
     * Otherwise, return NULL.
     * @return wrapped action model or NULL if none.
     */
    virtual ActionModel* getWrappedActionModel() const;

    /*!
     * Descend the wrapper chain to it's end.
     * Returns this action model if unwrap() returns NULL.
     * @return innermost action model
     */
    const ActionModel* unwrapAll() const;

    /*!
     * Descend the wrapper chain to it's end.
     * Returns this action model if unwrap() returns NULL.
     * @return innermost action model
     */
    ActionModel* unwrapAll();

    /*!
     * Find action model of a specific type in the wrapper chain.
     * @tparam AM type of action model to locate
     * @return typed action model or if not found.
     */
    template <typename AM>
    const AM* unwrap() const
    {
        const ActionModel* curr = this;
        // Loop through the chain
        while (true) {
            // Try to cast current action model
            auto cast = dynamic_cast<AM*>(curr);
            if (cast)
                return cast;

            // Obtain next wrapped action model
            const ActionModel* wrapped = curr->getWrappedActionModel();
            if (wrapped == NULL) {
                // end of chain
                return NULL;
            }
            // Descend
            curr = wrapped;
        }
    }

    /*!
     * Find action model of a specific type in the wrapper chain.
     * @tparam AM type of action model to locate
     * @return typed action model or if not found.
     */
    template <typename AM>
    AM* unwrap()
    {
        ActionModel* curr = this;
        // Loop through the chain
        while (true) {
            // Try to cast current action model
            auto cast = dynamic_cast<AM*>(curr);
            if (cast)
                return cast;

            // Obtain next wrapped action model
            ActionModel* wrapped = curr->getWrappedActionModel();
            if (wrapped == NULL) {
                // End of chain
                return NULL;
            }
            // Descend
            curr = wrapped;
        }
    }

protected:
    // Graph this action model operates on
    RcsGraph* graph;
};

} /* namespace Rcs */

#endif /* _ACTIONMODEL_H_ */
