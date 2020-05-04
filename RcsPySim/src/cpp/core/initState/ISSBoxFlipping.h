#ifndef _ISSBOXFLIPPING_H_
#define _ISSBOXFLIPPING_H_

#include "InitStateSetter.h"

namespace Rcs
{

/**
 * Initial state setter for the simplified box lifting task.
 * The initial state consists of the the x, y, and z positions of the end effector.
 */
class ISSBoxFlipping : public InitStateSetter
{
public:
    /**
     * Constructor.
     * The passed graph must contain the bodies Wrist1_L, Wrist2_L, Wrist3_L, Wrist1_R, Wrist2_R, and Wrist3_R.
     * @param graph graph to set the state on
     */
    ISSBoxFlipping(RcsGraph* graph);

    virtual ~ISSBoxFlipping();

    unsigned int getDim() const override;

    void getMinMax(double* min, double* max) const override;

    virtual std::vector<std::string> getNames() const;

    void applyInitialState(const MatNd* initialState) override;

private:
    RcsBody* wrist1L;
    RcsBody* wrist2L;
    RcsBody* wrist3L;
    RcsBody* wrist1R;
    RcsBody* wrist2R;
    RcsBody* wrist3R;
};

} /* namespace Rcs */

#endif /* _ISSBOXFLIPPING_H_ */
