#pragma once

#include "unicam/node_interface.hpp"

#include <MvCameraControl.h>

class HikVisionUsbCam : public CameraNodeInterface {
public:
    HikVisionUsbCam();
    ~HikVisionUsbCam() override final;

    bool is_alive() override final;
    void run() override final;

private:
    int device_id;
    void* handle;

    virtual rcl_interfaces::msg::SetParametersResult dynamic_reconfigure([[__attribute_maybe_unused__]] const std::vector<rclcpp::Parameter> &parameters) override final;

    static void image_callback(unsigned char *pData, MV_FRAME_OUT_INFO_EX *pFrameInfo, void *pUser);
    
};

