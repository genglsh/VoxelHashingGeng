/*
#include "ridOfPlane.h"
#include <Utils.h>
#include <math.h>
#include <random>

#define _USE_MATH_DEFINES

namespace ark{

    float gaussFunction(int x , int y, float covX = 100.f, float convY = 100.f ) {


        float index = (pow((x - 320), 2.0) + pow((y - 240), 2.0)) / float(320* 320 + 240*240);
//        printf("%f %f %f\n", index, float(1.0 / (2 * M_PI)), exp(-1 * index));
        printf("index is %f\n", index);
//        return  float(1.0 / (2 * M_PI * convY)) * pow(M_E, index);
        return index;
    }


    int ridOfPlane(const cv::Mat& image, Eigen::Vector4f& param){
        std::vector<Eigen::Vector3f> data;
        Eigen::Matrix3f cameraParam = Eigen::Matrix3f::Identity();
        float fx = 0.00193256, fy = 0.00193256, cx = -0.59026608, cy = -0.48393462;
        cameraParam(0, 0) = fx;
        cameraParam(0, 2) = cx;
        cameraParam(1, 1) = fy;
        cameraParam(1, 2) = cy;
//        Eigen::Matrix3f cameraParamInv = cameraParam.inverse();

        //设定一套采样规则，先稠密采样，再稀疏采样，再稠密采样。
        std::random_device rd;
        std::uniform_real_distribution<float> dist(0,1);

        for (int h = 0; h < 480; h += 4) {
            for (int w = 0; w < 640; w += 4) {

                if(1 || dist(rd) < ark::gaussFunction(w, h)) {
                    float depthV = (float) image.at<float>(h, w);
//                printf("%f\n", depthV);
                    if (depthV > 0 && depthV < 1000) {
                        Eigen::Vector3f axis(w, h, 1);
                        Eigen::Vector3f pcamera = depthV * cameraParam* axis;
                        data.emplace_back(pcamera[0], pcamera[1], pcamera[2]);
                    }
                }

            }
        }

        std::cout << data.size() << std::endl << std::endl;

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
        seg.setDistanceThreshold (PLANE_TRUNCATION_VALUE);
//        seg.setMaxIterations(100);

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
                        planeParam[2] * pcamera[2] + planeParam[3]) < PLANE_TRUNCATION_VALUE) {
                    cntElimate++;
                    image.at<float>(h, w) = 0.f;
                }
            }
        }
        return true;
    }
}


//int main() {
//    cv::Mat image = cv::imread("/home/gengshuai/Desktop/positive/test/new/VoxelHashingGeng_face/scene0220_02/depth/73.png", -1);
//    cv::Mat tem;
//    image.convertTo(tem, CV_32FC1);
//    Eigen::Vector4f planeParam;
//    ark::ridOfPlane(tem, planeParam);
//
//    ark::ridOfPlaneInDepth(tem, planeParam);
//    cv::imwrite("73RidPlane.png", tem);
//    return 0;
////    cv::Mat image(480, 640, CV_32FC1);
//
////    for (int h = 0; h < 480; h++) {
////        for (int w = 0; w < 640; w++) {
////
////            image.at<float>(h, w) =  ark::gaussFunction(w, h) * 225;
//////            printf("position %d %d is %f\n", h, w, image.at<float>(h, w));
////        }
////    }
//
//
////    cv::imwrite("2dGauss.png", image);
//
//}
*/

#include "ridOfPlane.h"
#include <Utils.h>
#include <math.h>
#include <random>
#include <pcl/filters/extract_indices.h>
//#define _USE_MATH_DEFINES

namespace ark{

    float gaussFunction(int x , int y, float covX = 100.f, float convY = 100.f ) {


        float index = (pow((x - 320), 2.0) + pow((y - 240), 2.0)) / float(320* 320 + 240*240);
//        printf("%f %f %f\n", index, float(1.0 / (2 * M_PI)), exp(-1 * index));
        printf("index is %f\n", index);
//        return  float(1.0 / (2 * M_PI * convY)) * pow(M_E, index);
        return index;
    }

    void Savetxt(pcl::PointCloud<pcl::PointXYZ>::Ptr pc1, const char* fileName) {
        ofstream fout(fileName);
        int num1 = pc1->size();

        for (int i = 0; i < num1; i++) {
            fout << pc1->at(i).x << " " << pc1->at(i).y << " " << pc1->at(i).z << "\n";
        }
    }


    int ridOfPlane(const cv::Mat& image, Eigen::Vector4f& param, int cnt, Eigen::Vector3f& frustumCenter){
        std::vector<Eigen::Vector3f> data;
        Eigen::Matrix3f cameraParam = Eigen::Matrix3f::Identity();
        float fx = 1.0 / ark::fx, fy = 1.0 / ark::fy , cx = -1.0 * ark::cx / ark::fx, cy = -1.0 * ark::cy / ark::fy;
        cameraParam(0, 0) = fx;
        cameraParam(0, 2) = cx;
        cameraParam(1, 1) = fy;
        cameraParam(1, 2) = cy;
//        Eigen::Matrix3f cameraParamInv = cameraParam.inverse();

        //设定一套采样规则，先稠密采样，再稀疏采样，再稠密采样。
        std::random_device rd;
        std::uniform_real_distribution<float> dist(0,1);

        for (int h = 0; h < 480; h += 4) {
            for (int w = 0; w < 640; w += 4) {

                if(1 || dist(rd) < ark::gaussFunction(w, h)) {
                    float depthV = (float) image.at<float>(h, w);
//                printf("%f\n", depthV);
                    if (depthV > 0 ) {
                        Eigen::Vector3f axis(w, h, 1);
                        Eigen::Vector3f pcamera = depthV * cameraParam* axis;
                        data.emplace_back(pcamera[0], pcamera[1], pcamera[2]);
                    }
                }

            }
        }

        std::cout << data.size() << std::endl << std::endl;

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
        seg.setDistanceThreshold (PLANE_TRUNCATION_VALUE);
        seg.setMaxIterations(100);

        ///** \brief Base method for segmentation of a model in a PointCloud given by <setInputCloud (), setIndices ()>
        seg.setInputCloud (cloud);

        seg.segment (*inliers, *coefficients);



//        int* index = (int*)malloc(sizeof(int) * cloud->points.size());
//        memset(index, 0 , sizeof(int) * cloud->points.size());
//        for(auto ele : inliers->indices){
//            index[ele] = 1;
//        }

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

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::PointCloud<pcl::PointXYZ>::Ptr outliner(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*outliner);
        float xSum = 0.f, ySum = 0.f, zSum = 0.f, xAvg = 0.f, yAvg = 0.f, zAvg = 0.f;
        for(const auto &ele : *outliner) {
            xSum += ele.x;
            ySum += ele.y;
            zSum += ele.z;
        }
        xAvg = xSum / outliner->size();
        yAvg = ySum / outliner->size();
        zAvg = zSum / outliner->size();
        printf(" xavg %f %f %f\n", xAvg, yAvg, zAvg);

        int xAxis = int(xAvg/zAvg * ark::fx + ark::cx);
        int yAxis = int(yAvg/zAvg * ark::fy + ark::cy);
        printf(" xAxis %d %d\n", xAxis, yAxis);

        frustumCenter[0] = xAxis, frustumCenter[1] = yAxis, frustumCenter[2] = zAvg;


        pcl::ModelCoefficients::Ptr coefficients2time (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers2time (new pcl::PointIndices);
        seg.setInputCloud (outliner);

        seg.segment (*inliers2time, *coefficients2time);
        printf("crash end\n");
        pcl::PointCloud<pcl::PointXYZ>::Ptr inlinerP(new pcl::PointCloud<pcl::PointXYZ>);

        extract.setInputCloud(outliner);
        extract.setIndices(inliers2time);
        extract.setNegative(false);
        extract.filter(*inlinerP);
        // 保存外点
//        printf("1\n");
//        std::string temName = "../scene0220_02/depth/" + std::to_string(cnt) + "Plane.txt";
//        ark::Savetxt( inlinerP, temName.data());
//        printf("2\n");
        return 0;
    }

    bool ridOfPlaneInDepth(cv::Mat& image, const Eigen::Vector4f& planeParam) {

        Eigen::Matrix3f cameraParam = Eigen::Matrix3f::Identity();
        float fx = 1.0 / ark::fx, fy = 1.0 / ark::fy , cx = -1.0 * ark::cx / ark::fx, cy = -1.0 * ark::cy / ark::fy;
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
                        planeParam[2] * pcamera[2] + planeParam[3]) < PLANE_TRUNCATION_VALUE) {
                    cntElimate++;
                    image.at<float>(h, w) = 0.f;
                }
            }
        }
        return true;
    }

    bool image2txt(cv::Mat& image, const Eigen::Matrix4f& c2w, int cnt) {

        Eigen::Matrix3f cameraParam = Eigen::Matrix3f::Identity();
        float fx = 1.0 / ark::fx, fy = 1.0 / ark::fy , cx = -1.0 * ark::cx / ark::fx, cy = -1.0 * ark::cy / ark::fy;
        cameraParam(0, 0) = fx;
        cameraParam(0, 2) = cx;
        cameraParam(1, 1) = fy;
        cameraParam(1, 2) = cy;
        ofstream fout (std::to_string(cnt) + ".txt");


//        Eigen::Matrix3f cameraParamInv = cameraParam.inverse();
        for (int h = 0; h < 480; h++) {
            for (int w = 0; w < 640; w++) {
                float depthV = (float) image.at<float>(h, w);
                if (depthV > 0 && depthV < 1000) {

                    Eigen::Vector3f axis(w, h, 1);
                    Eigen::Vector3f pcamera = depthV * cameraParam* axis;
                    Eigen::Vector4f tem(pcamera[0], pcamera[1], pcamera[2], 1);
                    auto tem4 = c2w * tem;
                    fout << tem4[0] << ' ' << tem4[1] << ' ' << tem4[2] << '\n';
                }
//                printf("%f\n", depthV);
            }
        }
        fout.close();
        return true;
    }
}

