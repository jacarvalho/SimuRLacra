#ifndef _OBSERVATIONMODEL_H_
#define _OBSERVATIONMODEL_H_

#include "../util/BoxSpaceProvider.h"

#include <Rcs_MatNd.h>
#include <Rcs_graph.h>

namespace Rcs
{

/**
 * The ObservationModel encapsulates the computation of the state vector from the current graph state.
 * This is used both for the observations returned to the policy as well as the state used to compute the reward.
 */
class ObservationModel : public BoxSpaceProvider
{
public:

    virtual ~ObservationModel();

    /**
     * Create a MatNd for the observation vector, fill it using computeObservation and return it.
     * The caller must take ownership of the returned matrix.
     * @param[in] currentAction action in current step. May be NULL if not available.
     * @param[in] dt time step since the last observation has been taken
     * @return a new observation vector
     */
    MatNd *computeObservation(const MatNd *currentAction, double dt) const;
    /**
     * Fill the given matrix with observation data.
     * @param[out] observation observation output vector
     * @param[in] currentAction action in current step. May be NULL if not available.
     * @param[in] dt time step since the last observation has been taken
     */
    void computeObservation(MatNd* observation, const MatNd *currentAction, double dt) const;

    /**
     * The number of state variables.
     */
    virtual unsigned int getStateDim() const = 0;

    /**
     * The number of velocity variables.
     * The default implementation assumes that for each state there is a velocity.
     */
    virtual unsigned int getVelocityDim() const;

    /**
     * Implement to fill the observation vector with the observed values.
     * @param[out] state state observation vector to fill, has getStateDim() elements.
     * @param[out] velocity velocity observation vector to fill, has getVelocityDim() elements.
     * @param[in] currentAction action in current step. May be NULL if not available.
     * @param[in] dt time step since the last observation has been taken
     */
    virtual void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const = 0;

    /**
     * Provides the minimum and maximum observable values.
     * Since the velocity is symmetric, only the maximum needs to be provided.
     * The default implementation uses -inf and inf.
     * @param[out] minState minimum state vector to fill, has getStateDim() elements.
     * @param[out] maxState maximum state vector to fill, has getStateDim() elements.
     * @param[out] maxVelocity maximum velocity vector to fill, has getVelocityDim() elements.
     */
    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;

    /**
     * Reset any internal state. This is called to begin a new episode.
     * It should also reset values depending on modifiable physics parameters.
     * This is an optional operation, so the default implementation does nothing.
     */
    virtual void reset();

    /**
     * Provides names for each state entry.
     * @return a vector of name strings. Must be of length getStateDim() or empty for a nameless space.
     */
    virtual std::vector<std::string> getStateNames() const;

    /**
     * Provides names for each velocity entry.
     * The default implementation derives the names from getStateNames(), appending a 'd' to each name.
     *
     * @return a vector of name strings. Must be of length getVelocityDim() or empty for a nameless space.
     */
    virtual std::vector<std::string> getVelocityNames() const;

    // These functions should not be overridden in subclasses!
    /**
     * Provides the minimum and maximum observable values.
     * Delegates to getLimits.
     */
    virtual void getMinMax(double* min, double* max) const final;

    /**
     * The number of observed variables is twice the number of state variables.
     * Delegates to getStateDim.
     */
    virtual unsigned int getDim() const final;

    /**
     * The velocity names are the state names postfixed with 'd'.
     * Delegates to getStateNames.
     */
    virtual std::vector<std::string> getNames() const final;

    /**
     * Find a nested observation model of a specified type.
     * If multiple observation models match, the first found in depth-first search order is returned.
     * @tparam OM observation model type to find
     * @return nested observation model or NULL if not found.
     */
    template <typename OM>
    OM* findModel() {
        auto dc = dynamic_cast<OM*>(this);
        if (dc)
            return dc;
        for (auto nested : getNested())
        {
            dc = nested->findModel<OM>();
            if (dc)
                return dc;
        }
        return NULL;
    }

    //! result of findOffsets
    struct Offsets
    {
        int pos;
        int vel;

        operator bool() const
        {
            return pos >= 0;
        }
    };

    /**
     * Find a nested observation model of a specified type.
     * If multiple observation models match, the first found in depth-first search order is returned.
     * NOTE: the positions and velovities are done separately. In order to correct for masked state observations use
     *       `observationModel->getStateDim() + thisOM.vel + i` to get the index.
     * @tparam OM observation model type to find
     * @return nested observation model or NULL if not found.
     */
    template <typename OM>
    Offsets findOffsets()
    {
        Offsets local = {0, 0};
        auto dc = dynamic_cast<OM*>(this);
        if (dc)
            return local;
        for (auto nested : getNested())
        {
            auto no = nested->findOffsets<OM>();
            if (no)
                return {local.pos + no.pos, local.vel + no.vel};

            local.pos += nested->getStateDim();
            local.vel += nested->getVelocityDim();
        }
        return {-1, -1 };
    }

    /**
     * List directly nested observation models.
     * The default implementation returns an empty list, since there are no nested models.
     */
    virtual std::vector<ObservationModel*> getNested() const;
};

} /* namespace Rcs */

#endif /* _OBSERVATIONMODEL_H_ */
