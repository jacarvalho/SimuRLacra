#ifndef RCSPYSIM_BOXSPACEPROVIDER_H
#define RCSPYSIM_BOXSPACEPROVIDER_H

#include "BoxSpace.h"
#include "nocopy.h"

namespace Rcs
{

/**
 * A class that lazily provides an 1D box space.
 */
class BoxSpaceProvider
{
private:
    mutable BoxSpace* space;

public:
    BoxSpaceProvider();

    virtual ~BoxSpaceProvider();

    // not copy- or movable
    RCSPYSIM_NOCOPY_NOMOVE(BoxSpaceProvider)

    /**
     * Compute and return the space.
     */
    const BoxSpace* getSpace() const;

    /**
     * Provides the number of elements in the space.
     * Since the BoxSpace object will be cached, this must not change.
     *
     * @return number of elements for the space.
     */
    virtual unsigned int getDim() const = 0;

    /**
     * Provides minimum and maximum values for the space.
     *
     * The passed arrays will be large enough to hold getDim() values.
     *
     * @param[out] min minimum value storage
     * @param[out] max maximum value storage
     */
    virtual void getMinMax(double* min, double* max) const = 0;

    /**
     * Provides names for each entry of the space.
     *
     * These are intended for use in python, i.e., for pandas dataframe column names.
     *
     * @return a vector of name strings. Must be of length getDim() or empty.
     */
    virtual std::vector<std::string> getNames() const;
};

} // namespace Rcs


#endif //RCSPYSIM_BOXSPACEPROVIDER_H
