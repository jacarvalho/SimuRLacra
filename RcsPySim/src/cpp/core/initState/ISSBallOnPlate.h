#ifndef _ISSBALLONPLATE_H_
#define _ISSBALLONPLATE_H_

#include "InitStateSetter.h"

namespace Rcs
{

/**
 * Initial state setter for the ball-on-plate environment.
 * The initial state consists of the x and y position of the ball relative to the plate center.
 */
    class ISSBallOnPlate : public InitStateSetter
    {
    public:
        /**
         * Constructor.
         * The passed graph must contain two bodies named "Ball" and "Plate".
         * @param graph graph to set the state on
         */
        ISSBallOnPlate(RcsGraph* graph);
        
        virtual ~ISSBallOnPlate();
        
        unsigned int getDim() const override;
        
        void getMinMax(double* min, double* max) const override;
        
        virtual std::vector<std::string> getNames() const;
        
        void applyInitialState(const MatNd* initialState) override;
    
    private:
        // The ball body
        RcsBody* ball;
        
        // The plate body
        RcsBody* plate;
        
        // The plate's dimensions
        double plateWidth;
        double plateHeight;
    };
    
} /* namespace Rcs */

#endif /* _ISSBALLONPLATE_H_ */
