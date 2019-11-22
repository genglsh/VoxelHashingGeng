//
// Created by gengshuai on 19-11-21.
//

#ifndef TSDF_RIDOFPLANE_H
#define TSDF_RIDOFPLANE_H

#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <thread>
#include <fstream>
#include<vector>
#include <sstream>


using namespace pcl;

namespace ark{
    int ridOfPlane(const cv::Mat& image, Eigen::Vector4f& param);

    bool ridOfPlaneInDepth(cv::Mat& image, const Eigen::Vector4f& param);
}

#endif //TSDF_RIDOFPLANE_H
