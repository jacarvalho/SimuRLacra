#ifndef RCSPYSIM_OMNORMALIZED_H
#define RCSPYSIM_OMNORMALIZED_H

#include "ObservationModel.h"
#include "../config/PropertySource.h"

namespace Rcs {

/*!
 * Observation model wrapper to normalize observations into the [-1,1] range.
 */
class OMNormalized : public ObservationModel {
private:
    // Wrapped observation model
    ObservationModel *wrapped;
    // Scale factor for every observation value. Contains both state and velocity.
    MatNd *scale;
    // Shift for every observation value, applied before scale. Contains both state and velocity.
    MatNd *shift;
public:
  /*!
   * Constructor
   * @param wrapped inner observation model. Takes ownership.
   * @param overrideMin overridden lower bounds by state name
   * @param overrideMax overridden upper bounds by state name
   */
    explicit OMNormalized(ObservationModel *wrapped, PropertySource* overrideMin, PropertySource* overrideMax);

    virtual ~OMNormalized();

    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(OMNormalized)

    virtual unsigned int getStateDim() const;

    virtual unsigned int getVelocityDim() const;

    virtual void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const;

    virtual void getLimits(double *minState, double *maxState, double *maxVelocity) const;

    virtual void reset();

    virtual std::vector<std::string> getStateNames() const;

    virtual std::vector<ObservationModel *> getNested() const;

    virtual std::vector<std::string> getVelocityNames() const;

};

} /* namespace Rcs */

#endif //RCSPYSIM_OMNORMALIZED_H
