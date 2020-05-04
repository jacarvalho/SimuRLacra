#ifndef _OMPARTIAL_H_
#define _OMPARTIAL_H_

#include "ObservationModel.h"

namespace Rcs
{

/**
 * Applies partial observability by masking out certain state variables.
 */
class OMPartial : public ObservationModel
{
public:
    typedef std::vector<unsigned int> IndexList;

    /**
     * Create a partial observation model using only the specified indices.
     * @param wrapped wrapped observation model. takes ownership.
     * @param indices selected indices. Applied to both state and velocity, assuming their dimensions are equal.
     * @param exclude set to true to invert the mask and keep only observations not in indices.
     */
    OMPartial(ObservationModel* wrapped, IndexList indices, bool exclude=false);

    /**
     * Create a partial observation model using only the specified indices.
     * @param wrapped wrapped observation model. takes ownership.
     * @param stateIndices selected indices for state.
     * @param velocityIndices selected indices for velocity.
     * @param exclude set to true to invert the mask and keep only observations not in indices.
     */
    OMPartial(ObservationModel* wrapped, IndexList stateIndices, IndexList velocityIndices, bool exclude=false);

    /**
     * Create a partial observation model using a boolean mask.
     * The mask is applied to both state and velocity, assuming their dimensions are equal.
     * @param wrapped wrapped observation model. takes ownership.
     * @param mask true entries will be kept.
     * @param exclude set to true to invert the mask
     */
    static OMPartial* fromMask(ObservationModel* wrapped, const std::vector<bool>& mask, bool exclude=false);

    /**
     * Create a partial observation model using a boolean mask.
     * @param wrapped wrapped observation model. takes ownership.
     * @param stateMask mask for state. Length must match wrapped->getStateDim().
     * @param velocityMask mask for velocity. Length must match wrapped->getVelocityDim().
     * @param exclude set to true to invert the mask
     */
    static OMPartial*
    fromMask(ObservationModel* wrapped, const std::vector<bool>& stateMask, const std::vector<bool>& velocityMask,
             bool exclude = false);

    virtual ~OMPartial();

    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(OMPartial)

    virtual unsigned int getStateDim() const;

    virtual unsigned int getVelocityDim() const;

    virtual void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const;

    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;

    virtual std::vector<std::string> getStateNames() const;

    virtual std::vector<std::string> getVelocityNames() const;

    virtual void reset();

    virtual std::vector<ObservationModel*> getNested() const;
private:

    // Wrapped full observation model
    ObservationModel* wrapped;

    // Vector of indices kept
    IndexList keptStateIndices;
    IndexList keptVelocityIndices;
};

} /* namespace Rcs */

#endif /* _OMPARTIAL_H_ */
