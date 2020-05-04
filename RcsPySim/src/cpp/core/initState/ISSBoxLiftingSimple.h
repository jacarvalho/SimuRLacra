#ifndef _ISSBOXLIFTINGSIMPLE_H_
#define _ISSBOXLIFTINGSIMPLE_H_

#include "InitStateSetter.h"

namespace Rcs
{

/**
 * Initial state setter for the simplified box lifting task.
 * The initial state consists of the the x, y, and z positions of the end effector.
 */
class ISSBoxLiftingSimple : public InitStateSetter
{
public:
    /**
     * Constructor.
     * The passed graph must contain the bodies Wrist1, Wrist2, Wrist3.
     * @param graph graph to set the state on
     */
    ISSBoxLiftingSimple(RcsGraph* graph);

    virtual ~ISSBoxLiftingSimple();

    unsigned int getDim() const override;

    void getMinMax(double* min, double* max) const override;

    virtual std::vector<std::string> getNames() const;

    void applyInitialState(const MatNd* initialState) override;

private:
    RcsBody* wrist1;
    RcsBody* wrist2;
    RcsBody* wrist3;
};

} /* namespace Rcs */

#endif /* _ISSBOXLIFTINGSIMPLE_H_ */
