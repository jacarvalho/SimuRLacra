#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <Rcs_MatNd.h>

namespace Rcs
{

/**
 * Integration mode.
 */
enum class IntMode
{
    ForwardEuler, // a.k.a. explicit Euler
    BackwardEuler, // a.k.a. implicit Euler
    SymplecticEuler
};

/*-----------------------------------------------------------------------------*
 * Notation
 *-----------------------------------------------------------------------------*/
// 0 indicates old/current time (begin of the integration interval)
// 1, 2, ..., T-1 indicate intermediate time steps
// T indicates the final time (end of the integration interval)

/**
 * Second order integration function.
 *
 * @param[in,out] x current value matrix, updated with new value.
 * @param[in,out] xd current first derivative value matrix, updated with new value.
 * @param[in] xdd second derivative matrix to integrate
 * @param dt timestep length in seconds
 * @param mode ForwardEuler or SymplecticEuler
 */
void intStep2ndOrder(MatNd* x, MatNd* xd, const MatNd* xdd,
                     double dt, IntMode mode);

/**
 * First order integration function.
 *
 * @param[in,out] x current value matrix, updated with new value.
 * @param[in] xd_0 first derivative matrix at current timestep to integrate. Used by ForwardEuler.
 * @param[in] xd_T first derivative matrix at next timestep to integrate. Used by BackwardEuler.
 * @param dt timestep length in seconds
 * @param mode ForwardEuler or BackwardEuler
 */
void intStep1stOrder(MatNd* x, const MatNd* xd_0, const MatNd* xd_T,
                     double dt, IntMode mode);

} // namespace Rcs

#endif // INTEGRATOR_H
