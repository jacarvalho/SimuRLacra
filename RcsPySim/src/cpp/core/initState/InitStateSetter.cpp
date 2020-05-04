#include "InitStateSetter.h"

#include <Rcs_VecNd.h>

#include <limits>

namespace Rcs
{

InitStateSetter::InitStateSetter(RcsGraph* graph) : graph(graph)
{
    // nothing else to do
}

// Set all min-values to -inf and all max-values to +inf by default
void InitStateSetter::getMinMax(double* min, double* max) const
{
    VecNd_setElementsTo(min, -std::numeric_limits<double>::infinity(), getDim());
    VecNd_setElementsTo(max, std::numeric_limits<double>::infinity(), getDim());
}

InitStateSetter::~InitStateSetter()
{
    // nothing to destroy
}

} /* namespace Rcs */
