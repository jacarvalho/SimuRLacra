#include "ObservationModel.h"

#include <Rcs_VecNd.h>
#include <Rcs_macros.h>

#include <limits>
#include <sstream>
#include <typeinfo>

namespace Rcs
{

ObservationModel::~ObservationModel() = default;

MatNd * ObservationModel::computeObservation(const MatNd *currentAction, double dt) const
{
    MatNd* result = MatNd_create(getDim(), 1);
    computeObservation(result, currentAction, dt);
    return result;
}

void ObservationModel::computeObservation(MatNd* observation, const MatNd *currentAction, double dt) const
{
    // First state, then velocity
    computeObservation(observation->ele, observation->ele + getStateDim(), currentAction, dt);
}

void ObservationModel::reset()
{
    // Do nothing
}

unsigned int ObservationModel::getVelocityDim() const
{
    // velocity dim == state dim
    return getStateDim();
}

// Set all min-values to -inf and all max-values to +inf by default
void ObservationModel::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    unsigned int sd = getStateDim();
    VecNd_setElementsTo(minState, -std::numeric_limits<double>::infinity(), sd);
    VecNd_setElementsTo(maxState, std::numeric_limits<double>::infinity(), sd);
    VecNd_setElementsTo(maxVelocity, std::numeric_limits<double>::infinity(), getVelocityDim());
}

std::vector<std::string> ObservationModel::getStateNames() const
{
    // Generate default names from class name and numbers
    const char* className = typeid(*this).name();

    std::vector<std::string> out;
    for (size_t i = 0; i < getStateDim(); ++i)
    {
        std::ostringstream os;
        os << className << "_" << i;
        out.push_back(os.str());
    }
    return out;
}

std::vector<std::string> ObservationModel::getVelocityNames() const
{
    // Fast track for no velocities case
    if (getVelocityDim() == 0)
        return {};
    RCHECK_MSG(getVelocityDim() == getStateDim(), "Must override getVelocityNames if velocity dim is not 0 or state dim.");
    
    // Append 'd' to each state name
    std::vector<std::string> out;
    for (auto& stateName : getStateNames()) {
        out.push_back(stateName + "d");
    }
    return out;
}


unsigned int ObservationModel::getDim() const
{
    // Observe state and velocity
    return getStateDim() + getVelocityDim();
}

void ObservationModel::getMinMax(double* min, double* max) const
{
    unsigned int sd = getStateDim();
    // Get the min and max velocity value pointer
    double* minVel = min + sd;
    double* maxVel = max + sd;

    // Obtain limits
    getLimits(min, max, maxVel);
    // Derive min velocity from max velocity
    VecNd_constMul(minVel, maxVel, -1, getVelocityDim());
}

std::vector<std::string> ObservationModel::getNames() const
{
    // concat state and velocity names
    auto res = getStateNames();
    auto vn = getVelocityNames();
    std::move(vn.begin(), vn.end(), std::inserter(res, res.end()));
    return res;
}

std::vector<ObservationModel*> ObservationModel::getNested() const
{
    return {};
}

} /* namespace Rcs */
