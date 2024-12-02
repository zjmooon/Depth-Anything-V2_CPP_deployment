#include "depth_anythingv2.h"

#include <iostream>
#include <fstream> 
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudawarping.hpp"

#include "yuv2rgb.cuh"

#include "pcl/ModelCoefficients.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/search/kdtree.h"
#include "pcl/segmentation/extract_clusters.h"

#include <thread>  
#include <chrono>  

#include "pcl/visualization/pcl_visualizer.h"
#include "pcl/io/pcd_io.h"
#include "pcl/sample_consensus/model_types.h"
#include "pcl/sample_consensus/method_types.h"

#include "cudaSegmentation.h"

extern int num;
#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
static cv::Rect roi(0, 228, 1920, 1080);

static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
}gLogger;

static void 
show_depth(const std::vector<float>& img_vector, int rows, int cols) {
    cv::Mat output = cv::Mat(rows, cols, CV_8UC1);
    float max_val = -1;
    float min_val = 9999999;
    for (int i = 0; i < img_vector.size(); i++) {
        max_val = std::max(max_val, img_vector[i]);
        min_val = std::min(min_val, img_vector[i]);
    }
    std::cout << __FILE__ << ":" << __LINE__ << "  max_val: " << max_val << std::endl;
    std::cout << __FILE__ << ":" << __LINE__ << "  min_val: " << min_val << std::endl;

    for (int i = 0; i < img_vector.size(); i++) {
        output.data[i] = ((img_vector[i] - min_val) / (max_val - min_val)) *255.0;
    }
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    cv::imshow("output", output);
    cv::waitKey(2000);
}
static void 
visual_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::PointIndices::Ptr &inliers, const pcl::ModelCoefficients::Ptr &coefficients)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler(cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color_handler, "original cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original cloud");

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*plane_cloud);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> plane_color_handler(plane_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(plane_cloud, plane_color_handler, "plane cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "plane cloud");

    // viewer->addPlane(*coefficients, "plane");
    viewer->addPlane(*coefficients, "plane");

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while(!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

static void 
CudaUndistort(cv::Mat &bgr_mat, cv::Mat &undist_bgr_mat, cv::Mat mtx, cv::Mat dist)
{
	cv::cuda::GpuMat src(bgr_mat);
	cv::cuda::GpuMat distortion(src.size(),src.type());

	cv::Size imageSize = src.size();

	cv::Mat map1, map2;
	initUndistortRectifyMap(
		mtx, dist, cv::Mat(),
		mtx, imageSize,
		CV_32FC1, map1, map2);

	cv::cuda::GpuMat m_mapx;
	cv::cuda::GpuMat m_mapy;
	m_mapx = cv::cuda::GpuMat(map1);
	m_mapy = cv::cuda::GpuMat(map2);
    
    // auto s = std::chrono::high_resolution_clock::now();
	cv::cuda::remap(src, distortion, m_mapx, m_mapy, cv::INTER_LINEAR);
	distortion.download(undist_bgr_mat);
    // auto e = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e - s);
    // std::cout << __FILE__ << ":"<<  __LINE__ << " ";
    // std::cout << "remap executed in " << duration.count() << " ms" << std::endl;
}

DepthAnythingV2::DepthAnythingV2(const Config &config)
{
    mConfig = config;
    std::string trt_path = mConfig.engine_path;
    /* Build the engine */ 
    std::ifstream file(trt_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << trt_path << " error!" << std::endl;
    }

    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serializedEngine = new char[size];
    file.read(serializedEngine, size);
    file.close();
    
    mRunTime = TRTUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    // mEngine.reset(mRunTime->deserializeCudaEngine(serializedEngine, size, nullptr));
    // mEngine = mRunTime->deserializeCudaEngine(serializedEngine, size);
    mEngine = TRTUniquePtr<nvinfer1::ICudaEngine>(mRunTime->deserializeCudaEngine(serializedEngine, size, nullptr));       
    // mContext = mEngine->createExecutionContext();
    mContext = TRTUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    delete[] serializedEngine;
    initialize();
}

void DepthAnythingV2::depthToPointCloud(const cv::Mat &depth_map, float fx, float fy, 
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &points) 
{
    points->clear();  // Clear the point cloud

    int width = depth_map.cols;
    int height = depth_map.rows;

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            float depth = depth_map.at<float>(v, u);  // Get the depth value at (u, v)

            if (depth > 0) {  // Process only valid depth values
                pcl::PointXYZ point;
                // Convert depth value to 3D point coordinates
                point.x = (u - width / 2.0f) * depth / fx;
                point.y = (v - height / 2.0f) * depth / fy;
                point.z = depth;

                points->points.push_back(point);  // Add the point to the point cloud
            }
        }
    }
    points->width = points->points.size();
    points->height = 1;
    points->is_dense = false;

    // std::cout << __FILE__ ":" << __LINE__ << " Point cloud data: " << points->size () << " points" << std::endl;
    // for (const auto& point: *points) {
    //     std::cout << "  " << point.x << " "
    //                         << point.y << " "
    //                         << point.z << std::endl;    }
}

void DepthAnythingV2::pclSegment(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<float> &equation) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>());

    std::vector<pcl::PointIndices> seg_indices;

    /* Create the segmentation object */
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);
    seg.setMaxIterations(50);
    seg.setProbability(0.9);
    seg.setInputCloud(cloud);

    // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // std::chrono::duration<double, std::ratio<1, 1000>> time_span =
    //     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
    // t1 = std::chrono::steady_clock::now();
    seg.segment(*inliers, *coefficients);
    // t2 = std::chrono::steady_clock::now();
    // time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
    // std::cout << "PCL(CPU) segment by Time: " << time_span.count() << " ms."<< std::endl;

    if (inliers->indices.size() == 0)
    {
        PCL_ERROR ("Could not estimate a planar model for the given dataset.\n");
        return ;
    }

    std::cout << __FILE__ ":" << __LINE__ << " Model coefficients: " << coefficients->values[0] << " " 
                                        << coefficients->values[1] << " "
                                        << coefficients->values[2] << " " 
                                        << coefficients->values[3] << "; Model inliers: " << inliers->indices.size () << std::endl;

    /* !!!!!!!!!!!!!!! visual point cloud !!!!!!!!!!!!!!! */ 
    // visual_cloud(cloud, inliers, coefficients);

    equation.emplace_back(coefficients->values[0]);
    equation.emplace_back(coefficients->values[1]);
    equation.emplace_back(coefficients->values[2]);
    equation.emplace_back(coefficients->values[3]);
}

void DepthAnythingV2::cudaSegment(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud){
    // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // std::chrono::duration<double, std::ratio<1, 1000>> time_span =
    // std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

    cudaStream_t stream = NULL;
    cudaStreamCreate (&stream);

    int nCount = cloud->width * cloud->height;
    float *inputData = (float *)cloud->points.data();

    float *input = NULL;
    cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
    cudaStreamAttachMemAsync (stream, input);
    //cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    int *index = NULL;
    // index should >= nCount of maximum inputdata,
    // index can be used for multi-inputs, be allocated and freed just at beginning and end
    cudaMallocManaged(&index, sizeof(int) * nCount, cudaMemAttachHost);
    cudaStreamAttachMemAsync (stream, index);
    cudaStreamSynchronize(stream);
    // modelCoefficients can be used for multi-inputs, be allocated and freed just at beginning and end
    float *modelCoefficients = NULL;
    int modelSize = 4;
    cudaMallocManaged(&modelCoefficients, sizeof(float) * modelSize, cudaMemAttachHost);
    cudaStreamAttachMemAsync (stream, modelCoefficients);
    cudaStreamSynchronize(stream);

    //Now Just support: SAC_RANSAC + SACMODEL_PLANE
    cudaSegmentation cudaSeg(SACMODEL_PLANE, SAC_RANSAC, stream);

    
    // t1 = std::chrono::steady_clock::now();
    segParam_t setP;
    setP.distanceThreshold = 0.1; 
    setP.maxIterations = 50;
    setP.probability = 0.8;
    setP.optimizeCoefficients = true;
    cudaSeg.set(setP);
    cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
    cudaSeg.segment(input, nCount, index, modelCoefficients);
    
    // std::vector<int> indexV;
    // for(int i = 0; i < nCount; i++)
    // {
    //     if(index[i] == 1) 
    //     indexV.push_back(i);
    // }

    // t2 = std::chrono::steady_clock::now();
    // time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
    // std::cout << "CUDA segment by Time: " << time_span.count() << " ms."<< std::endl;

    //std::cout << "CUDA index Size : " <<indexV.size()<< std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst(new pcl::PointCloud<pcl::PointXYZ>);
    cloudDst->width  = nCount;
    cloudDst->height = 1;
    cloudDst->points.resize (cloudDst->width * cloudDst->height);

//   pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNew(new pcl::PointCloud<pcl::PointXYZ>);
//   cloudNew->width  = nCount;
//   cloudNew->height = 1;
//   cloudNew->points.resize (cloudDst->width * cloudDst->height);

    int check = 0;
    for (std::size_t i = 0; i < nCount; ++i)
    {
    if (index[i] == 1)
    {
        cloudDst->points[i].x = input[i*4+0];
        cloudDst->points[i].y = input[i*4+1];
        cloudDst->points[i].z = input[i*4+2];
        check++;
    }
    // else if (index[i] != 1)
    // {
    //   cloudNew->points[i].x = input[i*4+0];
    //   cloudNew->points[i].y = input[i*4+1];
    //   cloudNew->points[i].z = input[i*4+2];
    // }
    }
//   pcl::io::savePCDFileASCII ("after-seg-cuda.pcd", *cloudDst);
//   pcl::io::savePCDFileASCII ("after-seg-cudaNew.pcd", *cloudNew);

    std::cout << __FILE__ ":" << __LINE__ << " CUDA modelCoefficients: " << modelCoefficients[0]
        <<" "<< modelCoefficients[1]
        <<" "<< modelCoefficients[2]
        <<" "<< modelCoefficients[3]
        <<" CUDA find points: " << check << std::endl;
    

    cudaFree(input);
    cudaFree(index);
    cudaFree(modelCoefficients);
}

void DepthAnythingV2::surfaceFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<float> &equation)
{
    pclSegment(cloud, equation);
    // cudaSegment(cloud);
}

bool DepthAnythingV2::surface_infer(void *img_buffer, std::size_t buffer_size, std::vector<float>& surface_equation)
{
    preInput(img_buffer, buffer_size);

    executeInfer();
    
    postOutput(surface_equation);

    std::cout << __FILE__ << ":" << __LINE__ << " ";
    for (auto x : surface_equation) {
        std::cout << x << ", ";
    }
    std::cout << std::endl;

    return true;
}

DepthAnythingV2::~DepthAnythingV2()
{
}

void DepthAnythingV2::preInput(void *img_buffer, std::size_t buffer_size)
{
    unsigned char* cuda_out_buffer = nullptr;

    if (buffer_size == 1920*1536*3) { // video input, BGR
        cuda_out_buffer = static_cast<unsigned char*>(img_buffer);  
    }
    else if (buffer_size == 1920*1536*2) { //camera input, YUYV
        cuda_out_buffer = (unsigned char*) malloc(1920*1536*3);
        gpuConvertYUYVtoBGR(static_cast<unsigned char*>(img_buffer), cuda_out_buffer, 1920, 1536);   
    }
    else {
        std::cerr << __FILE__ << ":" <<__LINE__ << ": " << std::endl << "Invalid void* buffer, please input 1920*1536*3 or 1920*1536*2 format data" << std::endl;
        return ;
    }

    cv::Mat bgr_mat(1536, 1920, CV_8UC3, cuda_out_buffer);
    cudaDeviceSynchronize();
    
    if (buffer_size == 1920*1536*2 && cuda_out_buffer != nullptr) {
        free(cuda_out_buffer);
        cuda_out_buffer = nullptr;
    }

    /* Undistort: CUDA boost */
    cv::Mat undist_bgr_mat;
    CudaUndistort(bgr_mat, undist_bgr_mat, mConfig.mtx, mConfig.dist);

    /* Crop to 1920x1080 */ 
    cv::Mat infer_mat = undist_bgr_mat(roi);

    /* Resize 924×518 */
    cv::Mat resized_img;
    cv::Size InferInputSize(mInput_w, mInput_h); 
    cv::resize(infer_mat, resized_img, InferInputSize, 0, 0, cv::INTER_CUBIC);
    if (num % 100 == 0) {
        std::string name = std::to_string(num) + ".jpg";
        cv::imwrite(name, resized_img);
    }

    /*char --> float, BGR --> RGB */
    resized_img.convertTo(resized_img, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);


    // HWC -> CHW  && normalize
    std::vector<float> chw(mInput_h * mInput_w * mInput_c);

    std::vector<double> mean_val = {0.485, 0.456, 0.406};
    std::vector<double> std_val = {0.229, 0.224, 0.225};
    for (int c=0; c < mInput_c; ++c) {
        for (int h = 0; h < mInput_h; ++h) {
            for (int w = 0; w < mInput_w; ++w) {
                chw[c * mInput_h * mInput_w + h * mInput_w + w] = (resized_img.at<cv::Vec3f>(h, w)[c] - mean_val[c]) / std_val[c];
            }
        }
    }
    
    CUDA_CHECK(cudaMemcpy(static_cast<float*>(mGpuBuffers[0]) , chw.data(), mInput_h*mInput_w*mInput_c*sizeof(float), cudaMemcpyHostToDevice));
    // cudaDeviceSynchronize();
}

bool DepthAnythingV2::executeInfer()
{
    // auto infer_start_time = std::chrono::high_resolution_clock::now();
    if (!mContext->executeV2(mGpuBuffers.data())) {
        std::cerr << "executeV2 error !!!" << std::endl;
        return false;
    }
    // auto infer_end_time = std::chrono::high_resolution_clock::now();
    // auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end_time - infer_start_time);
    // std::cout << __FILE__ << ":"<<  __LINE__ << " ";
    // std::cout << "infer executed in " << infer_duration.count() << " ms" << std::endl;
    
    return true; 
}

void DepthAnythingV2::postOutput(std::vector<float> &coefficients)
{
    size_t size = mOutputDims[0].d[0]*mOutputDims[0].d[1]*mOutputDims[0].d[2];
    std::vector<float> cpu_output(size);
    CUDA_CHECK(cudaMemcpy(cpu_output.data(), mGpuBuffers[1], size*sizeof(float), cudaMemcpyDeviceToHost));

    /* verify depth image */
    // show_depth(cpu_output, mOutputDims[0].d[1], mOutputDims[0].d[2]);
    // cv::Mat show_8u(mOutputDims[0].d[2], mOutputDims[0].d[1], CV_8UC1);
    // cv::normalize(original_depth, show_8u, 0, 255, cv::NORM_MINMAX, CV_8U);

    /* output depth width = mOutputDims[0].d[1],   output depth height = mOutputDims[0].d[2] */
    cv::Mat original_depth(mOutputDims[0].d[1], mOutputDims[0].d[2], CV_32FC1, cpu_output.data());
    

    cv::Mat resized_depth(1080, 1920, CV_32FC1);
    cv::resize(original_depth, resized_depth, cv::Size(1920, 1080), 0, 0, cv::INTER_LINEAR);

    /* crop image */
    cv::Mat cropped_depth = resized_depth(cv::Range(1080/2, 1080/2+200), cv::Range::all());

    cv::Mat show_8u(cropped_depth.rows, cropped_depth.cols, CV_8UC1);
    cv::normalize(cropped_depth, show_8u, 0, 255, cv::NORM_MINMAX, CV_8U);

    /* convert depth to pointcloud */
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    depthToPointCloud(cropped_depth, static_cast<float>(mConfig.mtx.at<double>(0,0)), 
                static_cast<float>(mConfig.mtx.at<double>(1,1)), point_cloud);

    /* surface fitting */
    surfaceFitting(point_cloud, coefficients);
}

void DepthAnythingV2::initialize()
{
    mGpuBuffers.resize(mEngine->getNbBindings());

    for (size_t i = 0; i < mEngine->getNbBindings(); ++i)
    {
        size_t binding_size = getSizeByDim(mEngine->getBindingDimensions(i));
        size_t binding_bytes = binding_size * sizeof(float);
        cudaMalloc(&mGpuBuffers[i], binding_bytes);

        if (mEngine->bindingIsInput(i))
        {
            mInputDims.emplace_back(mEngine->getBindingDimensions(i));
        }
        else
        {
            mOutputDims.emplace_back(mEngine->getBindingDimensions(i));
        }
    }
    if (mInputDims.empty() || mOutputDims.empty())
    {
        std::cerr << "Expect at least one input and one output for network";
        return ;
    }
    mInput_c = mInputDims[0].d[1];
    mInput_h = mInputDims[0].d[2]; 
    mInput_w = mInputDims[0].d[3];
    // std::cout <<  "mEngine->getNbBindings(): " <<  mEngine->getNbBindings() << std::endl;
    // std::cout << __FILE__ << ":"<<  __LINE__ << ":     mInput_w: " << mInput_w << ",mInput_h: " << mInput_h << std::endl;
    // std::cout << __FILE__ << ":"<<  __LINE__ << ":     mOutput_w: " << mOutputDims[0].d[1] << ",mOutput_h: " << mOutputDims[0].d[2] << std::endl;

}

size_t DepthAnythingV2::getSizeByDim(const nvinfer1::Dims &dims)
{
    size_t size = 1;
    
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }

    return size;
}
