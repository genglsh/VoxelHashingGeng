//
// Created by gengshuai on 19-8-28.
//

//# pragma once

#ifndef OPENARK_SLAMBASE_H
#define OPENARK_SLAMBASE_H

// 各种头文件
// C++标准库
#include <fstream>
#include <vector>
using namespace std;

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Utils.h>
#include<iostream>

#include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;

// OpenCV 特征检测模块
#include <opencv2/features2d/features2d.hpp>

//#include <opencv2/nonfree/nonfree.hpp> // use this if you want to use SIFT or SURF
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// 类型定义


//typedef pcl::PointXYZRGBA PointT;
//typedef pcl::PointCloud<PointT> PointCloud;

// 相机内参结构


// 函数接口
// image2PonitCloud 将rgb图转换为点云
//PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera );

// point2dTo3d 将单个点从图像坐标转换为空间坐标
// input: 3维点Point3f (u,v,d)
namespace ark{
    struct CAMERA_INTRINSIC_PARAMETERS
    {
        double cx, cy, fx, fy, scale;
    };

    class orbAlignment{
        public:
            orbAlignment();

            bool setCurrentFrame(const RGBDFrame& crtFrame,int FrameID);

            cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera );

            bool align();

            void ComputationalCharacteristics(vector<cv::KeyPoint>& kp, cv::Mat& des, const cv::Mat& rgb);

            RGBDFrame currentFrame;
            RGBDFrame lastFrame;
            cv::Mat lastRT;
            cv::Mat RT;
            int ID;
            cv::Ptr<cv::FeatureDetector> detector;
            cv::Ptr<cv::DescriptorExtractor> descriptor;
//            cv::Ptr<cv::Feature2D> detector;
    };
}
#endif //#define OPENARK_SLAMBASE_H



