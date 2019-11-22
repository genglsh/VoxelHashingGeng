//
// Created by gengshuai on 19-10-23.
//

#ifndef TSDF_ICP_PART_H
#define TSDF_ICP_PART_H

#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <string>
#include <sstream>
#include <zlib.h>
#include <unordered_map>
#include <pcl/common/eigen.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/search/kdtree.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <assert.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
using pcl::PointXYZ;
using pcl::PointCloud;

namespace ark {

    class ICPPart {
    public:

        //初始化参数，包括相机内参，最初始矩阵等等。
        ICPPart();

        bool align(Eigen::Matrix4f& final_transformation,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr tem,
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCVT,
                float& goal);

        bool CVTimage2Point(int frameID , const cv::Mat& image,
                            int steph = 8, int stepw = 8, int xS = 0,
                            int yS = 0, int xE = 640, int yE = 480);

        bool  CVTimage2Point(const cv::Mat& image, int steph = 4, int stepw = 4, int xS = 0,
                            int yS = 0);

        cv::Rect CVTimage2PointForeground(const cv::Mat& image, const float& depthV, int startW, int startH
                ,int steph = 4, int stepw = 4);

        bool CVTimage2PC(const cv::Mat& image, pcl::PointCloud<pcl::PointXYZ>::Ptr tem, int& numPC,
                int stepw = 8, int steph = 8);

        bool CalculateAlignment(pcl::PointCloud<pcl::PointXYZ>::Ptr tem,
                pcl::PointCloud<pcl::PointXYZ>::Ptr alignPC);

        Eigen::Matrix4f getRT();
        PointCloud<PointXYZ>::Ptr currentFrame;
        PointCloud<PointXYZ>::Ptr lastFrame;
        Eigen::Matrix3f cameraParam;
        Eigen::Matrix4f RT;
        Eigen::Matrix4f lastMat;
        int frameID;
    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    void SaveEignMat(const char* name, const Eigen::Matrix4f& init);

    cv::Mat GetDepth(int cnt);

    void open_Point_XYZ(const char* name, PointCloud<PointXYZ>::Ptr pc);

    void SavePLY(PointCloud<PointXYZ>::Ptr pc1, PointCloud<PointXYZ>::Ptr pc2, const char* fileName);

    void Savetxt(PointCloud<PointXYZ>::Ptr pc1, const char* fileName);

    void open_eigen_mat(const char* name, Eigen::Matrix4f& init);

    void getRidOfPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);


}

#endif //TSDF_ICP_PART_H
