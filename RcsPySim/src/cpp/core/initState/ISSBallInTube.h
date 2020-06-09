#ifndef _ISSBALLINTUBE_H_
#define _ISSBALLINTUBE_H_

#include "InitStateSetter.h"

namespace Rcs
{

/**
 * Initial state setter for the ball-in-tube task.
 * The initial state consists of the the x and y position of the base, and the z position of the rail.
 */
class ISSBallInTube : public InitStateSetter
{
public:
    /**
     * Constructor.
     * The passed graph must contain the bodies ImetronPlatform, RailBot.
     * @param graph graph to set the state on
     */
    ISSBallInTube(RcsGraph* graph);

    virtual ~ISSBallInTube();

    unsigned int getDim() const override;

    void getMinMax(double* min, double* max) const override;

    virtual std::vector<std::string> getNames() const;

    void applyInitialState(const MatNd* initialState) override;

private:
    RcsBody* platform;
    RcsBody* rail;
};

} /* namespace Rcs */

#endif /* _ISSBALLINTUBE_H_ */
