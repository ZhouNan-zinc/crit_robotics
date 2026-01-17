#pragma once

#include "unicam/node_interface.hpp"

#include <MvCameraControl.h>

/**
 * @brief Concrete camera driver that wraps the Hikrobot USB SDK.
 *
 * Discovers the requested USB device, configures exposure/gain parameters,
 * and publishes frames using the CameraNodeInterface utilities.
 */
class HikVisionUsbCam : public CameraNodeInterface {
public:
    HikVisionUsbCam();
    ~HikVisionUsbCam();

    /**
     * @brief Return whether the Hikrobot handle reports a live connection.
     */
    bool is_alive() override final;

    /**
     * @brief Enumerate devices, open the requested handle, and start streaming.
     */
    void run() override final;

private:
    int device_id;
    void* handle;

    /**
     * @brief Update camera parameters (exposure, gain, gamma) at runtime.
     */
    virtual rcl_interfaces::msg::SetParametersResult dynamic_reconfigure([[maybe_unused]] const std::vector<rclcpp::Parameter> &parameters) override final;

    /**
     * @brief Static callback invoked by the Hikrobot SDK on each frame.
     */
    static void image_callback(unsigned char *pData, MV_FRAME_OUT_INFO_EX *pFrameInfo, void *pUser);
    
};
