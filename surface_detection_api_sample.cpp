#include <iostream>
#include <fstream>
#include <filesystem>
#include "depth_anythingv2.h"
#include "opencv2/opencv.hpp"

int num=0;
unsigned char* readData(size_t& length, const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open file for reading." << std::endl;
        length = 0;
        return nullptr;
    }

    infile.seekg(0, std::ios::end);
    length = infile.tellg();
    infile.seekg(0, std::ios::beg);

    unsigned char* data = new unsigned char[length];
    infile.read(reinterpret_cast<char*>(data), length);
    infile.close();

    return data;
}

int main()
{
    /* SET CONFIG */
    Config config;
    config.debug_level = DEBUG_LEVEL::DEBUG_ALL;
    config.engine_path = "../model/dpAny2_metric_outdoor.trt";
    config.mtx = (cv::Mat_<double>(3, 3) << 
                                        1.0096978968416765e+03, 0., 9.9237751809854205e+02, 
                                        0., 1.0095734048948314e+03, 7.5946186432367506e+02, 
                                        0., 0., 1.);
    config.dist = (cv::Mat_<double>(1, 5) << 
                                        -3.6328291790539036e-01, 
                                        1.4643525547834677e-01, 
                                        -3.3306938293259716e-04, 
                                        7.7073209564316515e-04, 
                                        -2.7292745201340621e-02);

    DepthAnythingV2 dpAny2(config);

    size_t buffer_size;
    unsigned char* data;
    void* buffer;

    /* CAMERA INPUT */
    // for (const auto& entry : std::filesystem::directory_iterator("../infer_input/camera/")) {
    //     if (entry.is_regular_file() && entry.path().extension() == ".bin") { // Saving the raw data captured by the camera as a bin file in advance
    //         std::vector<float> equation;
    //         // std::cout << entry.path().string() << std::endl;
    //         data = readData(buffer_size, entry.path().string()); // If input is camera, the buffer_size=1920*1536*2 (YUYV), origin type is unsigned char*

    //         buffer = static_cast<void*>(data);
    //         dpAny2.surface_infer(buffer, buffer_size, equation);
    //     }
    // }

    /* VIDEO INPUT */
    cv::VideoCapture cap("../infer_input/video/031_20240229135140.mp4"); // Video requirements are: size 1920*1536 , uncalibrated
    if (!cap.isOpened() && config.debug_level>=DEBUG_LEVEL::DEBUG_ERROR) {
        std::cerr << "can not open the video" << std::endl;
        return -1;
    }

    cv::Mat frame;

    while(true) {
        num++;
        bool ret = cap.read(frame);
        if (!ret) {
            break;
        }

        data = frame.data;
        buffer = data;
        buffer_size = frame.total() * frame.elemSize(); // If input is video, the buffer_size=1920*1536*3 (BGR), origin type is unsigned char*

        std::vector<float> coefficients;
        
        // auto frame_start = std::chrono::high_resolution_clock::now();

        dpAny2.surface_infer(buffer, buffer_size, coefficients);

        // auto frame_end = std::chrono::high_resolution_clock::now();
        // auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
        // std::cout << "per frame spends " << frame_duration.count() << " ms" << std::endl;
    }

    return 0;
}