#include <RcsPyBot.h>
#include <config/PropertySourceXml.h>
#include <physics/PhysicsParameterManager.h>

#include <control/MLPPolicy.h>
#include <action/ActionModel.h>
#include <observation/ObservationModel.h>

#include <Rcs_macros.h>
#include <Rcs_utils.h>
#include <Rcs_utilsCPP.h>
#include <Rcs_timer.h>
#include <Rcs_parser.h>
#include <Rcs_resourcePath.h>
#include <Rcs_cmdLine.h>
#include <Rcs_mesh.h>

#include <KeyCatcherBase.h>
#include <HUD.h>

#include <ViewerComponent.h>
#include <PhysicsSimulationComponent.h>

#include <SegFaultHandler.h>



RCS_INSTALL_SEGFAULTHANDLER

bool runLoop = true;



/******************************************************************************
 * Ctrl-C destructor. Tries to quit gracefully with the first Ctrl-C
 * press, then just exits.
 *****************************************************************************/
static void quit(int /*sig*/)
{
    static int kHit = 0;
    runLoop = false;
    fprintf(stderr, "Trying to exit gracefully - %dst attempt\n", kHit + 1);
    kHit++;

    if (kHit == 2)
    {
        fprintf(stderr, "Exiting without cleanup\n");
        exit(0);
    }
}

int main(int argc, char** argv)
{
    Rcs::KeyCatcherBase::registerKey("q", "Quit");
    Rcs::KeyCatcherBase::registerKey("l", "Start/stop data logging");
    Rcs::KeyCatcherBase::registerKey("p", "Activate control policy");
    Rcs::KeyCatcherBase::registerKey("o", "Deactivate policy and return to initial state");

    RMSG("Starting Rcs...");
    char xmlFileName[128] = "exTargetTracking.xml", directory[128] = "config/TargetTracking";

    // Ctrl-C callback handler
    signal(SIGINT, quit);

    // This initialize the xml library and check potential mismatches between
    // the version it was compiled for and the actual shared library used.
    LIBXML_TEST_VERSION;

    // Parse command line arguments
    Rcs::CmdLineParser argP(argc, argv);
    argP.getArgument("-dl", &RcsLogLevel, "Debug level (default is 0)");
    argP.getArgument("-f", xmlFileName, "Configuration file name");
    argP.getArgument("-dir", directory, "Configuration file directory");
    bool valgrind = argP.hasArgument("-valgrind",
                                     "Start without Guis and graphics");
//    bool simpleGraphics = argP.hasArgument("-simpleGraphics", "OpenGL without "
//                                                         "fancy stuff (shadows, anti-aliasing)");

    runLoop = true;

    const char* hgr = getenv("SIT");
    if (hgr != NULL)
    {
        std::string meshDir = std::string(hgr) +
                              std::string("/Data/RobotMeshes/1.0/data");
        Rcs_addResourcePath(meshDir.c_str());
    }

    Rcs_addResourcePath("config");
    Rcs_addResourcePath(directory);

    // show help if requested
    if (argP.hasArgument("-h", "Show help message"))
    {
        Rcs::KeyCatcherBase::printRegisteredKeys();
        Rcs::CmdLineParser::print();
        Rcs_printResourcePath();
        return 0;
    }

    RMSG("Creating robot...");
    // create bot
    Rcs::RcsPyBot bot(new Rcs::PropertySourceXml(xmlFileName));

    // TODO add hardware components (which I don't have currently)
    // add physics sim for test
    Rcs::PhysicsParameterManager* ppmanager = bot.getConfig()->createPhysicsParameterManager();
    Rcs::PhysicsBase* simImpl = ppmanager->createSimulator(bot.getConfig()->properties->getChild("initdomainParam"));

    bot.getConfig()->actionModel->reset();
    bot.getConfig()->observationModel->reset();

    Rcs::PhysicsSimulationComponent* sim = new Rcs::PhysicsSimulationComponent(simImpl);
    sim->setUpdateFrequency(1.0 / bot.getConfig()->dt);
    bot.addHardwareComponent(sim);
    // and it does drive the update loop
    bot.setCallbackTriggerComponent(sim);

    // add viewer component
    Rcs::ViewerComponent* vc = NULL;
    Rcs::HUD* hud = NULL;
    if (!valgrind) {
        vc = new Rcs::ViewerComponent(bot.getGraph(), bot.getCurrentGraph(), true);
        hud = new Rcs::HUD(0, 0, 1024, 140);
        vc->getViewer()->add(hud);

        // experiment specific settings
        bot.getConfig()->initViewer(vc->getViewer());

        bot.addHardwareComponent(vc);
    }

    // load control policy
    Rcs::ControlPolicy* controlPolicy = NULL;
    auto policyConfig = bot.getConfig()->properties->getChild("policy");
    if (policyConfig->exists())
    {
        controlPolicy = Rcs::ControlPolicy::create(policyConfig);
    }

//    controlPolicy = Rcs::loadMLPPolicyFromXml(policyFile.c_str(),
//            bot.getConfig()->observationModel->getDim(), bot.getConfig()
//            ->actionModel->getDim());

    // start
    RMSG("Starting robot...");
    bot.startThreads();

    bool startLoggerNextPolicyStart = false;
    // main loop
    RMSG("Rcs is running!");
    while (runLoop) {
        // check keys
        if (vc && vc->getKeyCatcher()->getAndResetKey('q'))
        {
            runLoop = false;
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('l'))
        {
            if(bot.logger.isRunning()) {
                bot.logger.stop();
            } else if(bot.getControlPolicy() != controlPolicy) {
                // defer until policy start
                startLoggerNextPolicyStart = true;
                RMSG("Deferring logger start until the policy is activated");
            } else {

                bot.logger.start(bot.getConfig()->observationModel->getSpace(), bot.getConfig()->actionModel->getSpace(), 500);
            }
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('p'))
        {
            if(startLoggerNextPolicyStart) {
                bot.logger.start(bot.getConfig()->observationModel->getSpace(), bot.getConfig()->actionModel->getSpace(), 500);
            }
            bot.setControlPolicy(controlPolicy);
            RMSG("Control policy active.");
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('o'))
        {
            bot.setControlPolicy(NULL);
            RMSG("Holding initial state.");
        }
        if (vc && vc->getKeyCatcher()->getAndResetKey('b'))
        {
            // set ball position TODO
        }

        if(hud) {
            auto hudText = bot.getConfig()->getHUDText(simImpl->time(), bot.getObservation(), bot.getAction(), simImpl,
                                                       ppmanager, nullptr);
            hud->setText(hudText);
        }
        // wait a bit till next update
        Timer_waitDT(0.05);
    }

    // terminate
    RMSG("Terminating...");
    bot.stopThreads();
    bot.disconnectCallback();

    delete controlPolicy;

    delete ppmanager;

    // Clean up global stuff. From the libxml2 documentation:
    // WARNING: if your application is multithreaded or has plugin support
    // calling this may crash the application if another thread or a plugin is
    // still using libxml2. It's sometimes very hard to guess if libxml2 is in
    // use in the application, some libraries or plugins may use it without
    // notice. In case of doubt abstain from calling this function or do it just
    // before calling exit() to avoid leak reports from valgrind !
    xmlCleanupParser();

    fprintf(stderr, "Thanks for using the Rcs libraries\n");
    return 0;
}


