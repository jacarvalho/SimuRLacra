#include "OMPartial.h"

#include <Rcs_macros.h>

#include <algorithm>
#include <sstream>

namespace Rcs
{

using IndexList = OMPartial::IndexList;

// helpers for the constructor
static IndexList loadIndexList(IndexList input, unsigned int dim, bool exclude, const char* category)
{
    // verify
    for (auto idx : input)
    {
        if (idx >= dim)
        {
            std::ostringstream os;
            os << (exclude ? "Excluded " : "Selected ") << category << " index " << idx << " is outside of the value dimension "
               << dim;
            throw std::invalid_argument(os.str());
        }
    }

    // invert if needed
    if (exclude)
    {
        IndexList out;
        for (unsigned int idx = 0; idx < dim; ++idx)
        {
            // add if not in index list
            if (std::find(input.begin(), input.end(), idx) == input.end())
            {
                out.push_back(idx);
            }
        }
        return out;
    } else
    {
        return input;
    }
}

static IndexList loadMask(const std::vector<bool> &mask, unsigned int dim, bool exclude, const char* category)
{
    // verify
    if (mask.size() != dim)
    {
        std::ostringstream os;
        os << category << " mask size " << mask.size() << " does not match value dimension " << dim;
        throw std::invalid_argument(os.str());
    }
    // convert to index list
    IndexList out;
    for (unsigned int idx = 0; idx < dim; ++idx)
    {
        // add true entries if exclude is false, or false entries if exclude is true
        if (mask[idx] == !exclude)
        {
            out.push_back(idx);
        }
    }
    return out;
}

OMPartial::OMPartial(ObservationModel *wrapped,
                     IndexList indices, bool exclude) :
        wrapped(wrapped),
        keptStateIndices(loadIndexList(indices, wrapped->getStateDim(), exclude, "state"))
{
    if (wrapped->getVelocityDim() == wrapped->getStateDim()) {
        // use state for velocity
        keptVelocityIndices = keptStateIndices;
    } else if (wrapped->getVelocityDim() != 0) {
        // use explicit ctor
        throw std::invalid_argument("Cannot use same selection for state and velocity since their sizes don't match.");
    }
}


OMPartial::OMPartial(ObservationModel *wrapped, IndexList stateIndices, IndexList velocityIndices, bool exclude) :
        wrapped(wrapped),
        keptStateIndices(loadIndexList(stateIndices, wrapped->getStateDim(), exclude, "state")),
        keptVelocityIndices(loadIndexList(velocityIndices, wrapped->getVelocityDim(), exclude, "velocity"))
{
}

OMPartial *OMPartial::fromMask(ObservationModel *wrapped,
                               const std::vector<bool>& mask, bool exclude)
{
    return new OMPartial(wrapped,
                         loadMask(mask, wrapped->getStateDim(), exclude, "state"));
}

OMPartial *OMPartial::fromMask(ObservationModel *wrapped, const std::vector<bool>& stateMask, const std::vector<bool>& velocityMask,
                               bool exclude)
{
    return new OMPartial(wrapped,
                         loadMask(stateMask, wrapped->getStateDim(), exclude, "state"),
                         loadMask(velocityMask, wrapped->getVelocityDim(), exclude, "velocity"));
}

OMPartial::~OMPartial()
{
    delete wrapped;
}


// Fill partial with the selected values from full
template<typename T>
static void apply(T &partial, const T &full, const std::vector<unsigned int> &keptIndices)
{
    for (unsigned int i = 0; i < keptIndices.size(); ++i)
    {
        partial[i] = full[keptIndices[i]];
    }
}

unsigned int OMPartial::getStateDim() const
{
    return keptStateIndices.size();
}

unsigned int OMPartial::getVelocityDim() const
{
    return keptVelocityIndices.size();
}

void OMPartial::computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const
{
    // allocate temp storage for full, using matnd for potential stack optimization
    MatNd *state_full = NULL;
    MatNd *velocity_full = NULL;

    MatNd_create2(state_full, wrapped->getStateDim(), 1);
    MatNd_create2(velocity_full, std::max(wrapped->getVelocityDim(), 1u), 1);

    // retrieve from wrapped
    wrapped->computeObservation(state_full->ele, velocity_full->ele, currentAction, dt);

    // apply selection
    apply(state, state_full->ele, keptStateIndices);
    apply(velocity, velocity_full->ele, keptVelocityIndices);

    // clean up potential allocated memory
    MatNd_destroy(state_full);
    MatNd_destroy(velocity_full);
}

void OMPartial::getLimits(double *minState, double *maxState,
                          double *maxVelocity) const
{
    // allocate temp storage for full, using matnd for potential stack optimization
    MatNd *minState_full = NULL;
    MatNd_create2(minState_full, wrapped->getStateDim(), 1);
    MatNd *maxState_full = NULL;
    MatNd_create2(maxState_full, wrapped->getStateDim(), 1);
    MatNd *maxVelocity_full = NULL;
    MatNd_create2(maxVelocity_full, std::max(wrapped->getVelocityDim(), 1u), 1);

    // retrieve from wrapped
    wrapped->getLimits(minState_full->ele, maxState_full->ele, maxVelocity_full->ele);

    // apply selection
    apply(minState, minState_full->ele, keptStateIndices);
    apply(maxState, maxState_full->ele, keptStateIndices);
    apply(maxVelocity, maxVelocity_full->ele, keptVelocityIndices);

    // clean up potential allocated memory
    MatNd_destroy(minState_full);
    MatNd_destroy(maxState_full);
    MatNd_destroy(maxVelocity_full);
}

std::vector<std::string> OMPartial::getStateNames() const
{
    auto full = wrapped->getStateNames();
    std::vector<std::string> partial(keptStateIndices.size());
    apply(partial, full, keptStateIndices);
    return partial;
}

std::vector<std::string> OMPartial::getVelocityNames() const
{
    auto full = wrapped->getVelocityNames();
    std::vector<std::string> partial(keptVelocityIndices.size());
    apply(partial, full, keptVelocityIndices);
    return partial;
}

void OMPartial::reset()
{
    wrapped->reset();
}

std::vector<ObservationModel *> OMPartial::getNested() const
{
    return {wrapped};
}


} /* namespace Rcs */
