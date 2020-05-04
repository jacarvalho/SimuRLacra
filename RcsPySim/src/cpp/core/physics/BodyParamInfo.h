#ifndef _BODYPARAMINFO_H
#define _BODYPARAMINFO_H

#include <Rcs_graph.h>

#include <PhysicsConfig.h>

#include <string>

namespace Rcs
{

/**
 * Information on a body and changing physics params.
 *
 * Apart from the body, this struct holds material parameters not storable in RcsBody, and flags to track modified parameters.
 */
struct BodyParamInfo
{
    enum ModifiedFlag
    {
        // body mass changed
            MOD_MASS = 1 << 0,
        // body center of gravity changed
            MOD_COM = 1 << 1,
        // body inertia changed
            MOD_INERTIA = 1 << 2,
        // collision shapes changed
            MOD_SHAPE = 1 << 3,
        // position changed
            MOD_POSITION = 1 << 4
    };

    // The graph containing the body
    RcsGraph* graph;

    // The body
    RcsBody* body;

    // prefix for parameter names
    std::string paramNamePrefix;

    // body material - this is the material of the body's first shape.
    PhysicsMaterial material;

    // flags tracking the modified state of the body
    int modifiedFlag;

    BodyParamInfo(RcsGraph* graph, const char* bodyName, PhysicsConfig* physicsConfig);

    // reset all change flags
    void resetChanged();

    // test if a parameter changed
    inline bool isChanged(int flag)
    {
        return (modifiedFlag & flag) == flag;
    }


    // mark if a parameter as changed
    inline void markChanged(int flag)
    {
        modifiedFlag |= flag;
    }
};

} // namespace Rcs

#endif //_BODYPARAMINFO_H
