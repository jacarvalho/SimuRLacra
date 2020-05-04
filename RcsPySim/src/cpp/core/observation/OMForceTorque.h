#ifndef SRC_OBSERVATION_OMFORCETORQUE_H_
#define SRC_OBSERVATION_OMFORCETORQUE_H_

#include "OMComputedVelocity.h"

#include <limits>

namespace Rcs
{

/*!
 * Observes measurements of a single force/torque sensor.
 */
class OMForceTorque : public ObservationModel
{
public:

    /*!
     * Constructor
     * @param graph graph to observe
     * @param sensorName name of sensor to observe
     * @param maxForce maximum force in Newton for all sensed dimensions, e.g. 1200 N for the Kuka iiwa
     */
    OMForceTorque(RcsGraph* graph, const char* sensorName, double maxForce = std::numeric_limits<double>::infinity());

    virtual ~OMForceTorque();

    virtual unsigned int getStateDim() const;

    virtual unsigned int getVelocityDim() const;

    virtual void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const;
  
  virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;

    virtual std::vector<std::string> getStateNames() const;

private:
    // FTS to observe
    RcsSensor* sensor;
  
  double max_force;
  double max_torque;
};

} /* namespace Rcs */

#endif /* SRC_OBSERVATION_OMFORCETORQUE_H_ */
