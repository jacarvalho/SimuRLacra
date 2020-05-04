#ifndef SRC_CPP_CORE_DATALOGGER_H_
#define SRC_CPP_CORE_DATALOGGER_H_

#include "util/BoxSpace.h"

#include <Rcs_MatNd.h>

#include <string>
#include <mutex>
#include <fstream>

namespace Rcs
{

/**
 * Logs experiment data to a csv file.
 *
 * For every timestep, it records observations, actions and the reward.
 */
class DataLogger
{
public:
  
  /**
   * Constructor.
   *
   * @param fileBaseName base name for files, will append `filenum`.csv on start
   */
    DataLogger(std::string fileBaseName = "rollout_data_");
    virtual ~DataLogger();

    /**
     * Start logging for at most stepCount steps.
     *
     * @param[in] observationSpace environment's observation space
     * @param[in] actionSpace environment's action space
     * @param[in] maxStepCount maximum number of time steps
     * @param[in] filename filename to override default generated filename
     */
    void start(const BoxSpace* observationSpace, const BoxSpace* actionSpace, unsigned int maxStepCount,
        const char* filename=NULL);

    /**
     * Stop logging and flush data to file.
     */
    void stop();

    /**
     * Record data for the current step.
     */
    void record(const MatNd* observation, const MatNd* action, double reward);

    /**
     * Return true if running.
     */
    bool isRunning() const {
        return running;
    }
private:
    // update mutex
    std::recursive_mutex mutex;

    // auto log file naming
    std::string baseFileName;
    unsigned int fileCounter;

    // true if running
    volatile bool running;

    // buffer, avoids writing on realtime main thread
    MatNd* buffer;
    // step counter in current logging run
    unsigned int currentStep;

    // current output stream
    std::ofstream output;
};

} /* namespace Rcs */

#endif /* SRC_CPP_CORE_DATALOGGER_H_ */
