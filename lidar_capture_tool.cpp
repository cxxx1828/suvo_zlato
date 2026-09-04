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

'''
// The board is stationary during calibration -- instead of a single frame
// (which only covers a thin diagonal strip of the board due to how the
// Airy Lite scans), we accumulate several consecutive frames after the
// trigger and merge them into one dense point cloud.
static const int ACCUMULATE_FRAMES = 20;
static bool accumulating = false;
static int frames_collected = 0;
static std::string accumulating_index;
static pcl::PointCloud<PointT> accumulated_cloud;

void pointCloudPutCallback(std::shared_ptr<PointCloudMsg> msg)
{
    if (!accumulating)
    {
        // Save ONLY when the RGB script leaves a trigger -- otherwise we
        // would again get hundreds of .pcd files per second unrelated to
        // the RGB capture.
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
        std::remove(TRIGGER_FILE.c_str());  // consume the trigger immediately,
                                              // before accumulation starts
        accumulating = true;
        frames_collected = 0;
        accumulating_index = index;
        accumulated_cloud.points.clear();
        accumulated_cloud.height = 1;
    }
    # Add this frame's points to the accumulated cloud.
    accumulated_cloud.points.insert(accumulated_cloud.points.end(),
                                     msg->points.begin(), msg->points.end());
    frames_collected++;
    if (frames_collected < ACCUMULATE_FRAMES)
    {
        return;  // still accumulating
    }
    accumulated_cloud.width = static_cast<uint32_t>(accumulated_cloud.points.size());
    std::string filename = OUTPUT_DIR + "lidar_" + accumulating_index + ".pcd";
    try
    {
        pcl::io::savePCDFileASCII(filename, accumulated_cloud);
        std::cout << "Saved " << filename << " (" << accumulated_cloud.points.size()
                   << " pts from " << ACCUMULATE_FRAMES << " frames)\n";
    }
    catch (const pcl::IOException& e)
    {
        std::cerr << "Failed to save " << filename << ": " << e.what() << "\n";
    }
    accumulating = false;
}
'''
