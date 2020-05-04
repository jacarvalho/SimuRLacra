#include "OMNormalized.h"

#include <stdexcept>
#include <sstream>
#include <cmath>

static void validateAndOverride(MatNd* bound, Rcs::PropertySource* override, const char* boundName, const Rcs::BoxSpace* space) {
    // check each element
    auto& names = space->getNames();
    unsigned int nEle = bound->size;
    for (unsigned int i = 0; i < nEle; ++i) {
        auto bn = names[i];
        // try to load override
        override->getProperty(bound->ele[i], bn.c_str());
        // validate element is bounded now.
        if (std::isinf(bound->ele[i])) {
            std::ostringstream os;
            os << bn << " entry of " << boundName << " bound is infinite and not overridden. Cannot apply normalization.";
            throw std::invalid_argument(os.str());
        }
    }
}

Rcs::OMNormalized::OMNormalized(Rcs::ObservationModel *wrapped, PropertySource* overrideMin, PropertySource* overrideMax) : wrapped(wrapped)
{
    // get inner model bounds with optional overrides
    MatNd *iModMin = NULL;
    MatNd *iModMax = NULL;
    MatNd_clone2(iModMin, wrapped->getSpace()->getMin())
    MatNd_clone2(iModMax, wrapped->getSpace()->getMax())

    validateAndOverride(iModMin, overrideMin, "lower", wrapped->getSpace());
    validateAndOverride(iModMax, overrideMax, "upper", wrapped->getSpace());

    // Compute scale and shift from inner model bounds
    // shift is selected so that the median of min and max is 0
    // shift = min + (max - min)/2
    shift = MatNd_clone(iModMax);
    MatNd_subSelf(shift, iModMin);
    MatNd_constMulSelf(shift, 0.5);
    MatNd_addSelf(shift, iModMin);

    // scale = (max - min)/2
    scale = MatNd_clone(iModMax);
    MatNd_subSelf(scale, iModMin);
    MatNd_constMulSelf(scale, 0.5);

    // cleanup temporary matrices
    MatNd_destroy(iModMax);
    MatNd_destroy(iModMin);
}

Rcs::OMNormalized::~OMNormalized()
{
    // free matrices
    MatNd_destroy(shift);
    MatNd_destroy(scale);

    delete wrapped;
}

unsigned int Rcs::OMNormalized::getStateDim() const
{
    return wrapped->getStateDim();
}

unsigned int Rcs::OMNormalized::getVelocityDim() const
{
    return wrapped->getVelocityDim();
}

void Rcs::OMNormalized::computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const
{
    // query inner model
    wrapped->computeObservation(state, velocity, currentAction, dt);
    // normalize values
    unsigned int sdim = getStateDim();
    for (unsigned int i = 0; i < sdim; ++i)
    {
        state[i] = (state[i] - shift->ele[i]) / scale->ele[i];
    }
    for (unsigned int i = 0; i < getVelocityDim(); ++i)
    {
        velocity[i] = (velocity[i] - shift->ele[i + sdim]) / scale->ele[i + sdim];
    }
}

void Rcs::OMNormalized::getLimits(double *minState, double *maxState, double *maxVelocity) const
{
    // query inner model
    wrapped->getLimits(minState, maxState, maxVelocity);
    // report actual scaled bounds, not explicit overrides
    unsigned int sdim = getStateDim();
    for (unsigned int i = 0; i < sdim; ++i)
    {
        minState[i] = (minState[i] - shift->ele[i]) / scale->ele[i];
        maxState[i] = (maxState[i] - shift->ele[i]) / scale->ele[i];
    }
    for (unsigned int i = 0; i < getVelocityDim(); ++i)
    {
        maxVelocity[i] = (maxVelocity[i] - shift->ele[i + sdim]) / scale->ele[i + sdim];
    }
}

void Rcs::OMNormalized::reset()
{
    wrapped->reset();
}

std::vector<std::string> Rcs::OMNormalized::getStateNames() const
{
    return wrapped->getStateNames();
}

std::vector<std::string> Rcs::OMNormalized::getVelocityNames() const
{
    return wrapped->getVelocityNames();
}

std::vector<Rcs::ObservationModel *> Rcs::OMNormalized::getNested() const
{
    return {wrapped};
}
