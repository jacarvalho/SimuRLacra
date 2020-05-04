#ifndef _ISSPLANAR3LINK_H_
#define _ISSPLANAR3LINK_H_

#include "InitStateSetter.h"

namespace Rcs
{

/**
 * Initial state setter for the planar-3-link environment.
 * The initial state consists of the 3 joint angles.
 */
    class ISSPlanar3Link : public InitStateSetter
    {
    public:
        /**
         * Constructor.
         * The passed graph must contain three bodies named 'Link1', 'Link2', and 'Link3'.
         * @param graph graph to set the state on
         */
        ISSPlanar3Link(RcsGraph* graph);
        
        virtual ~ISSPlanar3Link();
        
        unsigned int getDim() const override;
        
        void getMinMax(double* min, double* max) const override;
        
        virtual std::vector<std::string> getNames() const;
        
        void applyInitialState(const MatNd* initialState) override;
    
    private:
        // The three link bodies
        RcsBody* link1;
        RcsBody* link2;
        RcsBody* link3;
    };
    
} /* namespace Rcs */

#endif /* _ISSPLANAR3LINK_H_ */
