#ifndef _ISSPLANARINSERT_H_
#define _ISSPLANARINSERT_H_

#include "InitStateSetter.h"

namespace Rcs
{

/**
 * Initial state setter for the planar-insert environment.
 * The initial state consists of the 4 joint angles.
 */
    class ISSPlanarInsert : public InitStateSetter
    {
    public:
        /**
         * Constructor.
         * The passed graph must contain five bodies named 'Link1', 'Link2', 'Link3', 'Link4', and 'Link5'.
         * @param graph graph to set the state on
         */
        ISSPlanarInsert(RcsGraph* graph);
        
        virtual ~ISSPlanarInsert();
        
        unsigned int getDim() const override;
        
        void getMinMax(double* min, double* max) const override;
        
        virtual std::vector<std::string> getNames() const;
        
        void applyInitialState(const MatNd* initialState) override;
    
    private:
        // The the link bodies
        RcsBody* link1;
        RcsBody* link2;
        RcsBody* link3;
        RcsBody* link4;
        RcsBody* link5;
    };
    
} /* namespace Rcs */

#endif /* _ISSPLANARINSERT_H_ */
