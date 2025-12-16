#include "unicam/hik_usb_cam.hpp"

#include <stdexcept>

HikVisionUsbCam::HikVisionUsbCam()
    : CameraNodeInterface(), device_id(0), handle(nullptr)
{
    if (int ecc = MV_CC_Initialize(); ecc != MV_OK) {
        RCLCPP_ERROR(logger, "HikVision SDK initialize failed. Check your SDK version and status.");
    }

    auto device = get_parameter_or<std::string>("device", "usb:0");
    try {
        auto pos = device.find(':');
        if (pos != std::string::npos) {
            if (auto device_type = device.substr(0, pos); device_type != "usb") {
                throw std::logic_error("Invalid device type: " + device_type);
            }
            device_id = std::stoi(device.substr(pos + 1));
        }
    } catch (std::exception &ex) {
        RCLCPP_ERROR_STREAM(get_logger(), "Invalid device format: `" << device << "` , assuming usb:0");
        device_id = 0;
    }

}

bool HikVisionUsbCam::is_alive() {
    return handle == nullptr ? false : MV_CC_IsDeviceConnected(handle);
}

void HikVisionUsbCam::run() {
    MV_CC_DEVICE_INFO_LIST device_info_list;
    MV_CC_EnumDevices(MV_USB_DEVICE, &device_info_list);

    if (handle != nullptr) {
        MV_CC_DestroyHandle(handle);
    }

    auto device_info = device_info_list.pDeviceInfo[device_id];
    MV_CC_CreateHandle(&handle, device_info);

    if (handle == nullptr) {
        throw std::runtime_error("Handle create failure");
    }

    MV_CC_RegisterImageCallBackEx(handle, &image_callback, this);

    if (MV_CC_IsDeviceConnected(handle)) {
        MV_CC_CloseDevice(handle);
    }

    MV_CC_SetExposureAutoMode(handle, MV_EXPOSURE_AUTO_MODE_OFF);
    MV_CC_SetGainMode(handle, MV_GAIN_MODE_OFF);
    MV_CC_SetPixelFormat(handle, PixelType_Gvsp_BGR8_Packed);
    MV_CC_SetTriggerMode(handle, MV_TRIGGER_MODE_OFF);
    MV_CC_SetAcquisitionMode(handle, MV_ACQ_MODE_CONTINUOUS);
    MV_CC_SetBalanceWhiteAuto(handle, MV_BALANCEWHITE_AUTO_ONCE);

    auto exposure_time = get_parameter_or<double>("exposure_time", 2500.);
    MV_CC_SetExposureTime(handle, exposure_time);
    auto gain = get_parameter_or<double>("gain", 15.);
    MV_CC_SetGain(handle, gain);
    auto gamma = get_parameter_or<double>("gamma", 1.0);
    MV_CC_SetGamma(handle, gamma);

    MV_CC_OpenDevice(handle, MV_ACCESS_Control);
    MV_CC_StartGrabbing(handle);
}

HikVisionUsbCam::~HikVisionUsbCam()
{
    if (handle != nullptr) {
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
    }

    if (int ecc = MV_CC_Finalize(); ecc != MV_OK) {
        RCLCPP_ERROR(logger, "HikVision SDK finalize failed with unknown error.");
    }
}

rcl_interfaces::msg::SetParametersResult HikVisionUsbCam::dynamic_reconfigure([[maybe_unused]] const std::vector<rclcpp::Parameter> &parameters)
{
    auto result = rcl_interfaces::msg::SetParametersResult().set__successful(true);

    if (handle == nullptr) {
        RCLCPP_WARN(logger, "Camera handle uncaptured, refusing setting parameter.");
        return result.set__successful(true);
    }

    for (const auto &param : parameters) {
        if (param.get_name() == "exposure_time") {
            auto exposure_time = param.as_double();
            MV_CC_SetExposureTime(handle, exposure_time);
        } else if (param.get_name() == "gain") {
            auto gain = param.as_double();
            MV_CC_SetGain(handle, gain);
        } else if (param.get_name() == "gamma") {
            auto gamma = param.as_double();
            MV_CC_SetGamma(handle, gamma);
        } else {
            RCLCPP_WARN_STREAM(logger, "Param `" << param.get_name() << "` does not support dynamic reconfigure");
            result.set__successful(false);
        }
    }

    return result;
}

void HikVisionUsbCam::image_callback(unsigned char *pData, MV_FRAME_OUT_INFO_EX *pFrameInfo, void *pUser)
{
    auto self = static_cast<HikVisionUsbCam *>(pUser);
    auto image = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3, static_cast<void *>(pData));
    self->publish(image);
}
