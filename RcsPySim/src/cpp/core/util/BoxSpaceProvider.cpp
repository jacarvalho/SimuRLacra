#include "BoxSpaceProvider.h"

Rcs::BoxSpaceProvider::BoxSpaceProvider() : space(NULL)
{}

Rcs::BoxSpaceProvider::~BoxSpaceProvider()
{
    delete space;
}

const Rcs::BoxSpace* Rcs::BoxSpaceProvider::getSpace() const
{
    if (!space)
    {
        // Create lazily
        MatNd* min = MatNd_create(getDim(), 1);
        MatNd* max = MatNd_create(getDim(), 1);

        getMinMax(min->ele, max->ele);

        space = new BoxSpace(min, max, getNames());
    }
    return space;
}

std::vector<std::string> Rcs::BoxSpaceProvider::getNames() const
{
    return {};
}
