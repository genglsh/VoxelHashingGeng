//
// Created by lucas on 1/28/18.
//

#ifndef OPENARK_UTILS_H
#define OPENARK_UTILS_H

#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <thread>
#include <queue>
#include <chrono>
#include <time.h>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <iostream>
#include <string>
#include <chrono>

using namespace std::chrono;
//using namespace std::string;

namespace ark{

    typedef pcl::PointXYZRGB PointType;

    void printTime(const system_clock::time_point &a,
                    const system_clock::time_point &b,
                    const std::string &note);

    class RGBDFrame {
    public:
        cv::Mat mTcw;
        cv::Mat imRGB;
        cv::Mat imDepth;
        int frameId;

        RGBDFrame(){
            mTcw = cv::Mat::eye(4,4,CV_32FC1);
            frameId = -1;
        }

        RGBDFrame(const RGBDFrame& frame)
        {
            frame.mTcw.copyTo(mTcw);
            frame.imRGB.copyTo(imRGB);
            frame.imDepth.copyTo(imDepth);
            frameId = frame.frameId;
        }

        RGBDFrame(const cv::Mat& imrgb, const cv::Mat& imdepth, int num) {
            //todo:应用在了ob模组的获取阶段，所以此处为了安全采用深拷贝（能否用浅拷贝减少时间?）
            imRGB = imrgb;
            imDepth = imdepth;
            frameId = num;
        }
        void setImRGB(const cv::Mat& imrgb){
            imRGB = imrgb;
        }

        void setImDepth(const cv::Mat& imdepth) {
            imDepth = imdepth;
        }

        void setFrameId(int num){
            frameId = num;
        }
        // 此处需要重载一个等号赋值函数，只copy深度图和rgb图。

//        void operator= ( RGBDFrame& tem){
//            this->imRGB = tem.imRGB.clone();
//            this->imDepth = tem.imDepth.clone();
//        }
    };
}

#endif //OPENARK_UTILS_H