#ifndef _ISSBOXSHELVING_H_
#define _ISSBOXSHELVING_H_

#include "InitStateSetter.h"

namespace Rcs
{

/**
 * Initial state setter for the box shelving task.
 * The initial state consists of the the x and y position of the base, the z position of the rail and two joint angles
 * of the left LBR arm.
 */
class ISSBoxShelving : public InitStateSetter
{
public:
    /**
     * Constructor.
     * The passed graph must contain the bodies ImetronPlatform, RailBot, lbr_link_2_L, lbr_link_4_L.
     * @param graph graph to set the state on
     */
    ISSBoxShelving(RcsGraph* graph);

    virtual ~ISSBoxShelving();

    unsigned int getDim() const override;

    void getMinMax(double* min, double* max) const override;

    virtual std::vector<std::string> getNames() const;

    void applyInitialState(const MatNd* initialState) override;

private:
    RcsBody* platform;
    RcsBody* rail;
    RcsBody* link2L;
    RcsBody* link4L;
};

} /* namespace Rcs */

#endif /* _ISSBOXSHELVING_H_ */
