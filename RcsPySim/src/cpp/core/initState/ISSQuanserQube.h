#ifndef INITSTATE_ISSQUANSERQUBE_H_
#define INITSTATE_ISSQUANSERQUBE_H_

#include "InitStateSetter.h"

namespace Rcs
{

class ISSQuanserQube : public InitStateSetter
{
public:
    ISSQuanserQube(RcsGraph* graph);

    virtual ~ISSQuanserQube();

    unsigned int getDim() const override;

    void getMinMax(double* min, double* max) const override;

    void applyInitialState(const MatNd* initialState) override;

private:
    // Body for the actuated pole
    RcsBody* arm;
    // Body for the pendulum pole
    RcsBody* pendulum;
};

} /* namespace Rcs */

#endif /* INITSTATE_ISSQUANSERQUBE_H_ */
