#include <catch2/catch.hpp>

#include <config/PropertySourceXml.h>
#include <RcsSimEnv.h>

#include <Rcs_macros.h>
#include <Rcs_resourcePath.h>

using namespace Rcs;

TEST_CASE("Environment run")
{
    // Set Rcs debug level
    RcsLogLevel = 2;
    
    // Make sure the resource path is set up
    Rcs_addResourcePath("config");
    
    std::vector<std::string> configs{"config/BallOnPlate/exBotKuka.xml", "config/TargetTracking/exTargetTracking.xml"};
    
    for (auto& configFile : configs)
    {
        DYNAMIC_SECTION("Config " << configFile)
        {
            RcsSimEnv env(new PropertySourceXml(configFile.c_str()));
            
            // Reset env
            MatNd* obs = env.reset(PropertySource::empty(), NULL);
            
            // Verify observation
            REQUIRE(env.observationSpace()->checkDimension(obs));
            MatNd_destroy(obs);
            
            MatNd* action = env.actionSpace()->createValueMatrix();
            
            // Perform random steps
            for (int step = 0; step < 100; ++step)
            {
                // Make a random action
                env.actionSpace()->sample(action);
                
                // Perform step
                obs = env.step(action);
                
                // Cannot really verify observation, an observation outside the space is valid and leads to termination.
                MatNd_destroy(obs);
            }
            
            MatNd_destroy(action);
        }
    }
}
