
#include <memory>  // Must come before the rs_driver headers

#include <rs_driver/api/lidar_driver.hpp>
#include <rs_driver/msg/pcl_point_cloud_msg.hpp>  // defines robosense::lidar::PointCloudT
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <string>
#include <chrono>
#include <thread>       // this_thread::sleep_for -- explicit to avoid the previous include bug
#include <fstream>
#include <iostream>
#include <cstdio>       // std::remove
#include <stdexcept>
#include <unistd.h>     // readlink
#include <linux/limits.h> // PATH_MAX

using namespace robosense::lidar;

typedef pcl::PointXYZI PointT;
typedef PointCloudT<PointT> PointCloudMsg;

// --- Automatic project root directory detection ---
// The executable always resides in <project_root>/build/lidar_capture_tool,
// so derive <project_root> reliably from /proc/self/exe regardless of
// where the tool was launched. This permanently resolves the cwd dependency
// that previously caused pcl::IOException.
std::string getExecutableDir()
{
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len == -1)
    {
        throw std::runtime_error("Cannot read /proc/self/exe");
    }
    buf[len] = '\0';
    std::string exe_path(buf);
    return exe_path.substr(0, exe_path.find_last_of('/'));
}

static const std::string EXE_DIR      = getExecutableDir();       // .../lidar_calibration/build
static const std::string PROJECT_ROOT = EXE_DIR + "/..";           // .../lidar_calibration
static const std::string OUTPUT_DIR   = PROJECT_ROOT + "/calib_images/lidar/";
static const std::string TRIGGER_FILE = OUTPUT_DIR + ".trigger";

std::shared_ptr<PointCloudMsg> pointCloudGetCallback()
{
    return std::make_shared<PointCloudMsg>();
}

void pointCloudPutCallback(std::shared_ptr<PointCloudMsg> msg)
{
    // Save ONLY when the RGB script leaves a trigger -- otherwise we would
    // again get hundreds of .pcd files per second unrelated to the RGB capture.
    std::ifstream trigger(TRIGGER_FILE);
    if (!trigger.good())
    {
        return;  // no trigger; skip this frame
    }

    std::string index;
    trigger >> index;
    trigger.close();

    if (index.empty())
    {
        return;  // trigger file is (still) empty -- wait for the next frame
    }

    std::string filename = OUTPUT_DIR + "lidar_" + index + ".pcd";
    try
    {
        pcl::io::savePCDFileASCII(filename, *msg);
        std::cout << "Saved " << filename << "\n";
    }
    catch (const pcl::IOException& e)
    {
        std::cerr << "Failed to save " << filename << ": " << e.what() << "\n";
    }

    std::remove(TRIGGER_FILE.c_str());  // trigger has been consumed
}

void exceptionCallback(const Error& code)
{
    // Log and continue.
}

int main()
{
    LidarDriver<PointCloudMsg> driver;
    RSDriverParam param;
    param.lidar_type = LidarType::RSAIRYLITE_ETH;
    param.input_type = InputType::ONLINE_LIDAR;
    param.input_param.msop_port = 6699;
    param.input_param.difop_port = 7788;

    driver.regPointCloudCallback(pointCloudGetCallback, pointCloudPutCallback);
    driver.regExceptionCallback(exceptionCallback);

    if (!driver.init(param))
    {
        std::cerr << "Driver init failed\n";
        return -1;
    }
    driver.start();

    std::cout << "Output dir: "   << OUTPUT_DIR   << "\n";
    std::cout << "Trigger file: " << TRIGGER_FILE  << "\n";
    std::cout << "Waiting for triggers, Ctrl+C to stop...\n";
    while (true) { std::this_thread::sleep_for(std::chrono::seconds(1)); }
}
