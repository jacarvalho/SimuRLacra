#ifndef _ISSBOXLIFTING_H_
#define _ISSBOXLIFTING_H_

#include "InitStateSetter.h"

namespace Rcs
{

/**
 * Initial state setter for the box lifting task.
 * The initial state consists of the the x and y position of the base, the z position of the rail and two joint angles
 * of the left and the right LBR arm.
 */
class ISSBoxLifting : public InitStateSetter
{
public:
    /**
     * Constructor.
     * The passed graph must contain the bodies ImetronPlatform, RailBot, lbr_link_2_L, lbr_link_2_R.
     * @param graph graph to set the state on
     */
    ISSBoxLifting(RcsGraph* graph);

    virtual ~ISSBoxLifting();

    unsigned int getDim() const override;

    void getMinMax(double* min, double* max) const override;

    virtual std::vector<std::string> getNames() const;

    void applyInitialState(const MatNd* initialState) override;

private:
    RcsBody* platform;
    RcsBody* rail;
    RcsBody* link2L;
    RcsBody* link2R;
};

} /* namespace Rcs */

#endif /* _ISSBOXLIFTING_H_ */
