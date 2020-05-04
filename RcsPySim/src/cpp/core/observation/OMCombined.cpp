#include "OMCombined.h"

namespace Rcs
{

OMCombined::~OMCombined()
{
    // delete parts
    for (auto part : parts) {
        delete part;
    }
}

void OMCombined::addPart(ObservationModel* part)
{
    parts.push_back(part);
}

unsigned int Rcs::OMCombined::getStateDim() const
{
    // do it for all parts
    unsigned int sumdim = 0;
    for (auto part : parts) {
        sumdim += part->getStateDim();
    }
    return sumdim;
}

unsigned int OMCombined::getVelocityDim() const
{
    // do it for all parts
    unsigned int sumdim = 0;
    for (auto part : parts) {
        sumdim += part->getVelocityDim();
    }
    return sumdim;
}

void Rcs::OMCombined::computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const
{
    // do it for all parts
    for (auto part : parts) {
        part->computeObservation(state, velocity, currentAction, dt);
        state += part->getStateDim();
        velocity += part->getVelocityDim();
    }
}

void Rcs::OMCombined::getLimits(double* minState, double* maxState,
        double* maxVelocity) const
{
    // do it for all parts
    for (auto part : parts) {
        part->getLimits(minState, maxState, maxVelocity);
        minState += part->getStateDim();
        maxState += part->getStateDim();
        maxVelocity += part->getVelocityDim();
    }
}

std::vector<std::string> Rcs::OMCombined::getStateNames() const
{
    // reserve dim out vars
    std::vector<std::string> out;
    out.reserve(getStateDim());
    // concatenate names from parts
    for (auto part : parts) {
        auto pnames = part->getStateNames();
        // move the elements from pnames since it is a copy anyways.
        std::move(pnames.begin(), pnames.end(), std::inserter(out, out.end()));
    }
    return out;
}

std::vector<std::string> OMCombined::getVelocityNames() const
{
    // reserve dim out vars
    std::vector<std::string> out;
    out.reserve(getVelocityDim());
    // concatenate names from parts
    for (auto part : parts) {
        auto pnames = part->getVelocityNames();
        // move the elements from pnames since it is a copy anyways.
        std::move(pnames.begin(), pnames.end(), std::inserter(out, out.end()));
    }
    return out;
}

void OMCombined::reset()
{
    // do it for all parts
    for (auto part : parts) {
        part->reset();
    }
}

std::vector<ObservationModel*> OMCombined::getNested() const
{
    return parts;
}

} /* namespace Rcs */
