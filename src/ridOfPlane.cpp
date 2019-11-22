#include "ridOfPlane.h"

namespace ark{
    int ridOfPlane(const cv::Mat& image, Eigen::Vector4f& param){
        std::vector<Eigen::Vector3f> data;
        Eigen::Matrix3f cameraParam = Eigen::Matrix3f::Identity();
        float fx = 0.00193256, fy = 0.00193256, cx = -0.59026608, cy = -0.48393462;
        cameraParam(0, 0) = fx;
        cameraParam(0, 2) = cx;
        cameraParam(1, 1) = fy;
        cameraParam(1, 2) = cy;
//        Eigen::Matrix3f cameraParamInv = cameraParam.inverse();
        for (int h = 0; h < 480; h += 4) {
            for (int w = 0; w < 640; w += 4) {
                float depthV = (float) image.at<float>(h, w);
//                printf("%f\n", depthV);
                if (depthV > 0 && depthV < 1000) {
                    Eigen::Vector3f axis(w, h, 1);
                    Eigen::Vector3f pcamera = depthV * cameraParam* axis;
                    data.emplace_back(pcamera[0], pcamera[1], pcamera[2]);
                }
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        cloud->width  = data.size();
        cloud->height = 1;
        cloud->points.resize (cloud->width * cloud->height);

        for(int i = 0; i < cloud->width; i++) {
            cloud->points[i].x = data[i][0];
            cloud->points[i].y = data[i][1];
            cloud->points[i].z = data[i][2];
        }

        std::cerr << "Point cloud data: " << cloud->points.size () << " points" << std::endl;

        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
//        pcl::PCLPointCloud2ConstPtr
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients (true);
        // Mandatory
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold (5);

        ///** \brief Base method for segmentation of a model in a PointCloud given by <setInputCloud (), setIndices ()>
        seg.setInputCloud (cloud);

        seg.segment (*inliers, *coefficients);
        printf("crash end\n");
//
//        if (inliers->indices.size () == 0)
//        {
//            PCL_ERROR ("Could not estimate a planar model for the given dataset.");
//            return 0;
//        }

        std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                  << coefficients->values[1] << " "
                  << coefficients->values[2] << " "
                  << coefficients->values[3] << std::endl;
        for(int x = 0; x < 4; x++)
            param[x] = coefficients->values[x];
        return 0;
    }

    bool ridOfPlaneInDepth(cv::Mat& image, const Eigen::Vector4f& planeParam) {

        Eigen::Matrix3f cameraParam = Eigen::Matrix3f::Identity();
        float fx = 0.00193256, fy = 0.00193256, cx = -0.59026608, cy = -0.48393462;
        cameraParam(0, 0) = fx;
        cameraParam(0, 2) = cx;
        cameraParam(1, 1) = fy;
        cameraParam(1, 2) = cy;
        int cntElimate = 0;
//        Eigen::Matrix3f cameraParamInv = cameraParam.inverse();
        for (int h = 0; h < 480; h++) {
            for (int w = 0; w < 640; w++) {
                float depthV = (float) image.at<float>(h, w);
//                printf("%f\n", depthV);
                Eigen::Vector3f axis(w, h, 1);
                Eigen::Vector3f pcamera = depthV * cameraParam* axis;
                if(fabs(pcamera[0] * planeParam[0] + planeParam[1] * pcamera[1] +
                        planeParam[2] * pcamera[2] + planeParam[3]) < 5) {
                    cntElimate++;
                    image.at<float>(h, w) = 0.f;
                }
            }
        }
        return true;
    }
}


//int main(){
//    cv::Mat image = cv::imread("/home/gengshuai/Desktop/positive/test/new/VoxelHashingGeng face/scene0220_02/depth/73.png", -1);
//    cv::Mat tem;
//    image.convertTo(tem, CV_32FC1);
//    Eigen::Vector4f planeParam;
//    ark::ridOfPlane(tem, planeParam);
//
//    ark::ridOfPlaneInDepth(tem, planeParam);
//    cv::imwrite("73RidPlane.png", tem);
//    return 0;
//}