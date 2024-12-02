#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "opencv2/core.hpp"
#include "NvInfer.h"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"


enum DEBUG_LEVEL
{
    DEBUG_NONE = 0, // print nothing
    DEBUG_ERROR = 1,  // print error
    DEBUG_ALL = 2  // print all
};

typedef struct Config
{   
    std::string engine_path;
    enum DEBUG_LEVEL debug_level;  

    /* calibration */
    cv::Mat mtx;
    cv::Mat dist;

    Config() : debug_level(DEBUG_LEVEL::DEBUG_NONE),
        mtx(cv::Mat::zeros(3, 3, CV_64F)), dist(cv::Mat::zeros(1, 5, CV_64F)) { }
} Config;

struct TRTDestroy {
    template<class T> 
    void operator()(T* obj) const {
        obj->destroy();
    }
};

class DepthAnythingV2
{
template< class T >
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

private:
    TRTUniquePtr<nvinfer1::IRuntime> mRunTime;
    TRTUniquePtr<nvinfer1::ICudaEngine> mEngine;
    TRTUniquePtr<nvinfer1::IExecutionContext> mContext;

    std::vector<nvinfer1::Dims> mInputDims;            //! The dimensions of the input to the network.
    std::vector<nvinfer1::Dims> mOutputDims;           //! The dimensions of the output to the network.
    std::vector<void*> mGpuBuffers;                    //! The vector of device buffers needed for engine execution
    int mInput_w;                                      //! The width of the image required by the network
    int mInput_h;                                      //! The height of the image required by the network
    int mInput_c;                                      //! The channel of the image required by the network
    struct Config mConfig;

public:
    DepthAnythingV2(const Config &config);
    bool surface_infer(void* img_buffer, std::size_t buffer_size, std::vector<float>& surface_equation);
    ~DepthAnythingV2();

private:
    void initialize();
	void preInput(void* img_buffer, std::size_t buffer_size);  // void * --> (924*518 ) ...
    bool executeInfer();
	void postOutput(std::vector<float> &coefficients);
    size_t getSizeByDim(const nvinfer1::Dims& dims);
    void depthToPointCloud(const cv::Mat &depth_map, float fx, float fy, pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud);
    void surfaceFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<float> &equation);
    void pclSegment(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<float> &equation);
    void cudaSegment(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
};
