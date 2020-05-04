#ifndef _OMCOMBINED_H_
#define _OMCOMBINED_H_

#include "ObservationModel.h"

#include <vector>

namespace Rcs
{

/**
 * Combines multiple ObservationModels into one by concatenating their observation vectors.
 */
class OMCombined : public ObservationModel
{
public:
    virtual ~OMCombined();

    /**
     * Add an ObservationModel as part of this combined model.
     *
     * This must be done right after construction, before reset() is called for the first time.
     *
     * OMCombined will assume ownership of the passed object.
     *
     * @param[in] part part model to add
     */
    void addPart(ObservationModel* part);

    virtual unsigned int getStateDim() const;

    virtual unsigned int getVelocityDim() const;

    virtual void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const;

    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;

    virtual void reset();

    virtual std::vector<std::string> getStateNames() const;

    virtual std::vector<std::string> getVelocityNames() const;

    virtual std::vector<ObservationModel*> getNested() const;

private:
    std::vector<ObservationModel*> parts;
};

} /* namespace Rcs */

#endif /* _OMCOMBINED_H_ */
