#include "SimpleLogger.h"
#include "doctest.h"

using namespace sipai;

TEST_CASE("Testing SimpleLogger") {

  auto &logger = SimpleLogger::getInstance();

  SUBCASE("Test precision") {
    auto currentPrecision = logger.getPrecision();
    CHECK(currentPrecision > 0);
    logger.setPrecision(currentPrecision + 1);
    CHECK(logger.getPrecision() == currentPrecision + 1);
    logger.setPrecision(currentPrecision);
  }

  SUBCASE("Test messages") {
    std::ostringstream ossout;
    std::ostringstream osserr;
    logger.setStreamOut(ossout);
    logger.setStreamErr(osserr);

    logger.log(LogLevel::INFO, true, "Test message");
    MESSAGE(ossout.str());
    CHECK(ossout.str().find("[INFO] Test message") != std::string::npos);
    ossout.str("");

    logger.log(LogLevel::WARN, true, "Test message2");
    CHECK(osserr.str().find("[WARN] Test message2") != std::string::npos);
    osserr.str("");

    logger.log(LogLevel::ERROR, true, "Test message3");
    CHECK(osserr.str().find("[ERROR] Test message3") != std::string::npos);
    osserr.str("");

    logger.log(LogLevel::DEBUG, true, "Test message4");
    CHECK(osserr.str().find("[DEBUG] Test message4") != std::string::npos);
    osserr.str("");

    logger.log((LogLevel)100, true, "Test message5");
    CHECK(osserr.str().find("[UNKNOWN] Test message5") != std::string::npos);
    osserr.str("");

    SimpleLogger::LOG_INFO("Test message6");
    CHECK(ossout.str().find("[INFO] Test message6") != std::string::npos);
    ossout.str("");

    SimpleLogger::LOG_WARN("Test message7");
    CHECK(osserr.str().find("[WARN] Test message7") != std::string::npos);
    osserr.str("");

    SimpleLogger::LOG_ERROR("Test message8");
    CHECK(osserr.str().find("[ERROR] Test message8") != std::string::npos);
    osserr.str("");

    SimpleLogger::LOG_DEBUG("Test message9");
    CHECK(osserr.str().find("[DEBUG] Test message9") != std::string::npos);
    osserr.str("");

    // Restore std::cout
    logger.setStreamOut(std::cout);
    logger.setStreamErr(std::cerr);
  }
}