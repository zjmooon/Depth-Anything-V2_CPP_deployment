## Description

### Road surface detection module
1. Use the monocular [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2) model to get the depth value of the input image.
2. Combine the depth value and the camera intrinsic parameters to get the point cloud.

3. Use pcl::SACMODEL_PLANE + pcl::SAC_RANSAC to fit the road surface and output array {A, B, C, D} (Ax+By+Cz+D=0).

### API
> bool surface_infer(void* img_buffer, std::size_t buffer_size, std::vector<float>& surface_equation);


The buffer can come from the camera or the video:
- The data type of the raw data needs to be unsigned char*. 
- The size of the raw data should be 1920*1536 and not calibrated.

### NOTE:
 Currently only support two data format, the API has been adapted internally
-  camera: buffer_size=1920\*1536\*2 (YUYV)
-  video :  buffer_size=1920\*1536\*3 (BGR)

## Dependencies
``` shell
sudo apt install libpcl-dev 
```

## Build
``` shell 
mkdir build
cd build
cmake ..
make
```

## Run

Usage:<br>
```
./surface_detection
```


**Test Enviroment**
```
Jetson AGX Orin 32GB
Jetpack 5.1.2
CUDA 11.4
OpenCV with CUDA 4.4.0
PCL 1.10
```



