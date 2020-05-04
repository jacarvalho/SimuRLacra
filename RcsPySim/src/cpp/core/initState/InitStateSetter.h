#ifndef INITSTATESETTER_H_
#define INITSTATESETTER_H_

#include "../util/BoxSpaceProvider.h"

#include <Rcs_graph.h>

namespace Rcs
{

/**
 * The InitStateSetter defines the changeable initial state of the simulation.
 * It is invoked during the reset() method to adapt the initial state as desired for the rollout.
 */
    class InitStateSetter : public BoxSpaceProvider
    {
    public:
        /**
         * Constructor
         * @param graph graph to set the state on
         */
        explicit InitStateSetter(RcsGraph* graph);
        
        virtual ~InitStateSetter();
        
        /**
         * Provides the minimum and maximum state values.
         * The default implementation uses -inf and inf.
         */
        virtual void getMinMax(double* min, double* max) const;
        
        /**
         * Set initial state of the graph.
         */
        virtual void applyInitialState(const MatNd* initialState) = 0;
    
    protected:
        // the graph
        RcsGraph* graph;
    };
    
} /* namespace Rcs */

#endif /* INITSTATESETTER_H_ */
