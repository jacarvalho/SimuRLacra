#ifndef _ISSMPBLENDING_H_
#define _ISSMPBLENDING_H_

#include "InitStateSetter.h"

namespace Rcs
{

/**
 * Initial state setter for the movement primitive blending sandbox environment.
 * The initial state consists of the 2 Cartesian positions of the prismatic joints.
 */
    class ISSMPBlending : public InitStateSetter
    {
    public:
        /**
         * Constructor.
         * The passed graph must contain a body named 'Effector'.
         * @param graph graph to set the state on
         */
        ISSMPBlending(RcsGraph* graph);
        
        virtual ~ISSMPBlending();
        
        unsigned int getDim() const override;
        
        void getMinMax(double* min, double* max) const override;
        
        virtual std::vector<std::string> getNames() const;
        
        void applyInitialState(const MatNd* initialState) override;
    
    private:
        // The only moving body
        RcsBody* effector;
    };
    
} /* namespace Rcs */

#endif /* _ISSMPBLENDING_H_ */
