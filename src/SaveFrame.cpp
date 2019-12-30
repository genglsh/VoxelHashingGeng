//
// Created by yiwen on 2/2/19.
//

#include <chrono>
#include <mutex>
#include <Utils.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/cudaimg>
//#include <opencv/>

//#include <MathUtils.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/filters/fast_bilateral.h>
// #include <opencv2/ximgproc.hpp>
#include <opencv2/opencv.hpp>
#include "SaveFrame.h"

namespace ark {

    void RenderDepthMap(const cv::Mat depthRaw, cv::Mat& depthImg){

        const static int MAX_DEPTH_VALUE = 0xffff;
        int width = 640, height = 480;
        float* pDepthHist = new float[MAX_DEPTH_VALUE];
        memset(pDepthHist, 0, MAX_DEPTH_VALUE * sizeof(float));

        int numberOfPoints = 0;
        unsigned short nvalue = 0;
        for (int row = 0; row < height; row++){
            for (int col = 0; col < width; col++){
                nvalue = depthRaw.at<unsigned short>(row, col);
                if (nvalue != 0){
                    pDepthHist[nvalue] ++;
                    numberOfPoints ++;
                }
            }
        }

        for (int i = 1; i < MAX_DEPTH_VALUE; i ++) {
            pDepthHist[i] += pDepthHist[i - 1];
        }

        for (int i = 1; i < MAX_DEPTH_VALUE; i ++) {
            if (pDepthHist[i] != 0) {
                pDepthHist[i] = (numberOfPoints - pDepthHist[i]) / (float)numberOfPoints;
            }
        }


        for (int row = 0; row < height; row++) {
            uchar * showcell = (uchar *)depthImg.ptr<uchar>(row);
            for (int col = 0; col < width; col++)
            {
                char depthValue = pDepthHist[depthRaw.at<unsigned short>(row, col)] * 255;
                *showcell++ = 0;
                *showcell++ = depthValue;
                *showcell++ = depthValue;
            }
        }
    }

    void createFolder(struct stat &info, std::string folderPath){
        if(stat( folderPath.c_str(), &info ) != 0 ) {
            std::cout<< "Error:"<< folderPath<<" doesn't exist!" << std::endl;
            exit(1);

            if (-1 == mkdir(folderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
            {
                std::cout<< "Error creating directory "<< folderPath<<" !" << std::endl;
                exit(1);
            }
            std::cout << folderPath << " is created" << folderPath << std::endl;
        }else if( info.st_mode & S_IFDIR )  // S_ISDIR() doesn't exist on my windows
            std::cout<<folderPath<<" is a directory"<<std::endl;
        else
            std::cout<<folderPath<<" is no directory"<<std::endl;
    }

    SaveFrame::SaveFrame(std::string folderPath) {

        struct stat info;

        createFolder(info, folderPath);

        rgbPath = folderPath +"RGB/";
        depthPath = folderPath +"depth/";
        tcwPath = folderPath +"tcw/";

        createFolder(info, rgbPath);
        createFolder(info, depthPath);
        createFolder(info, tcwPath);

        mKeyFrame.frameId = -1;
        mbRequestStop = false;
    }

    void SaveFrame::Start() {
        mptRun = new std::thread(&SaveFrame::Run, this);
    }


    void SaveFrame::RequestStop() {
        std::unique_lock<std::mutex> lock(mRequestStopMutex);
        mbRequestStop = true;
    }

    bool SaveFrame::IsRunning() {
        std::unique_lock<std::mutex> lock(mRequestStopMutex);
        return mbRequestStop;
    }

    void SaveFrame::Run() {
//        ark::RGBDFrame currentKeyFrame;
//        while (true) {
//            {
//                std::unique_lock<std::mutex> lock(mRequestStopMutex);
//                if (mbRequestStop)
//                    break;
//            }
//
//
//            {
//                std::unique_lock<std::mutex> lock(mKeyFrameMutex);
//                if (currentKeyFrame.frameId == mKeyFrame.frameId)
//                    continue;
//                mKeyFrame.imDepth.copyTo(currentKeyFrame.imDepth);
//                mKeyFrame.imRGB.copyTo(currentKeyFrame.imRGB);
//                mKeyFrame.mTcw.copyTo(currentKeyFrame.mTcw);
//                currentKeyFrame.frameId = mKeyFrame.frameId;
//            }
//
//            cv::Mat Twc = mKeyFrame.mTcw.inv();
//
////            Reproject(currentKeyFrame.imRGB, currentKeyFrame.imDepth, Twc);
//        }
    }

    void SaveFrame::OnKeyFrameAvailable(const RGBDFrame &keyFrame) {
        if (mMapRGBDFrame.find(keyFrame.frameId) != mMapRGBDFrame.end())
            return;
        std::cout << "OnKeyFrameAvailable" << keyFrame.frameId << std::endl;
        // std::unique_lock<std::mutex> lock(mKeyFrameMutex);
        keyFrame.mTcw.copyTo(mKeyFrame.mTcw);
        keyFrame.imRGB.copyTo(mKeyFrame.imRGB);
        keyFrame.imDepth.copyTo(mKeyFrame.imDepth);

        mKeyFrame.frameId = keyFrame.frameId;
        mMapRGBDFrame[keyFrame.frameId] = ark::RGBDFrame();
    }

    void SaveFrame::OnFrameAvailable(const RGBDFrame &frame) {
        std::cout << "OnFrameAvailable" << frame.frameId << std::endl;
    }

    void SaveFrame::OnLoopClosureDetected() {
        std::cout << "LoopClosureDetected" << std::endl;
    }

    void SaveFrame::frameWrite(const RGBDFrame &frame){
        if (mMapRGBDFrame.find(frame.frameId) != mMapRGBDFrame.end())
            return;

        std::cout<<"frameWrite frame = "<<frame.frameId<<std::endl;
        if(frame.frameId > 300)
            return;

//        std::unique_lock<std::mutex> lock(mKeyFrameMutex);
        cv::imwrite(rgbPath + std::to_string(frame.frameId) + ".png", frame.imRGB);

        cv::Mat depth255;
        //cv::normalize(frame.imDepth, depth255, 0, 1000, cv::NORM_MINMAX, CV_16UC1); ////cast to 16

        
        frame.imDepth.convertTo(depth255, CV_16UC1, 1000);
        cv::imwrite(depthPath + std::to_string(frame.frameId) + ".png", depth255);

        cv::FileStorage fs(tcwPath + std::to_string(frame.frameId)+".xml",cv::FileStorage::WRITE);
        fs << "tcw" << frame.mTcw ;
        //fs << "depth" << frame.imDepth ;
        fs.release();

        /*
        cv::FileStorage fs2(depth_to_tcw_Path + std::to_string(frame.frameId)+".xml",cv::FileStorage::WRITE);
        fs2 << "depth" << depth255;
        // fs << "rgb" << frame.imRGB;
        fs2.release();
        */

        mMapRGBDFrame[frame.frameId] = ark::RGBDFrame();

    }

    RGBDFrame SaveFrame::frameLoad(int frameId){
        std::cout<<"frameLoad frame ==================== "<<frameId<<std::endl;
        RGBDFrame frame;

        frame.frameId = frameId;


        cv::Mat rgbBig = cv::imread(rgbPath + std::to_string(frame.frameId) + ".png",cv::IMREAD_COLOR);
//        std::cout << rgbBig.size().width << " " << rgbBig.size().height << " " << rgbBig.channels() << std::endl;

        if(rgbBig.rows == 0){
            frame.frameId = -1;
            return frame;
        }

        cv::resize(rgbBig, frame.imRGB, cv::Size(640,480));
        std::cout << frame.imRGB.rows << " rgb "<<frame.imRGB.cols <<std::endl;
        std::cout<<depthPath + std::to_string(frame.frameId) + ".png"<<std::endl;
        cv::Mat depth255 = cv::imread(depthPath + std::to_string(frame.frameId) + ".png",-1);


        //当前参数设置情况下，未发现双边滤波有什么效果。
        cv::Mat depth255tem, bilaterRes(640, 480, CV_32FC1), depth8u(640, 480, CV_16UC1);

        depth255.convertTo(depth255tem, CV_32FC1);

        cv::bilateralFilter(depth255tem, bilaterRes, 5, 20, 20);
//       cv::cuda::bilateralFilter
//        std::cout << "begin" << std::endl;
//        cv::Mat renderRes(480, 640, CV_16UC3);
//
//        bilaterRes.convertTo(depth8u, CV_16UC1);
////        std::cout << "2" << std::endl;
//        RenderDepthMap(depth8u, renderRes);
////        std::cout << "3" << std::endl;
//        cv::imshow("shungbain", renderRes);
//
//        std::cout << "end" << std::endl;
//        cv::waitKey();
//        cv::cuda::bilateralFilter
//        depth255.convertTo(frame.imDepth, CV_32FC1);

//        for(int h = 0; h < 480; h++) {
//
//            for(int w = 0; w < 640; w++) {
//
//                if(abs(depth255tem.at<float>(h, w) - bilaterRes.at<float>(h, w)) > 0)
////                    std::cout << "change! " << std::endl;
//                    assert(false);
//            }
//
//        }
        bilaterRes.convertTo(frame.imDepth, CV_32FC1);
        // 当前设置下的深度范围为 0-10 对应我们的最大范围。
        //frame.imDepth *= 0.056;
//        frame.imDepth *= 0.0222;
//        frame.imDepth *= 0.01667;
//        frame.imDepth *= 1.0;
        //         std::cout << "depth255 = "<< std::endl << " "  << frame.imDepth << std::endl << std::endl;
        // return;
        /*
            这个位置的超参有待商榷,将深度进行了一个整体压缩,需要根据实际情况调整压缩幅度.
        */
        //cv::normalize(depth255, frame.imDepth, 0.2, 10, cv::NORM_MINMAX, CV_32F);

        //TCW FROM XML
        /*
        cv::FileStorage fs2(tcwPath + std::to_string(frame.frameId)+".xml", cv::FileStorage::READ);
        fs2["tcw"] >> frame.mTcw;
        
        //fs2["depth"] >> frame.imDepth;
        fs2.release();
        */

        //TCW FROM TEXT

        // * 从文件中读取旋转矩阵
        double tcwArr[4][4];
        std::ifstream tcwFile;
        tcwFile.open(tcwPath + std::to_string(frame.frameId) + ".txt");
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 4; ++k) {
                tcwFile >> tcwArr[i][k];
            }
        }
        cv::Mat tcw(4, 4, CV_64FC1, tcwArr);

        frame.mTcw = tcw;

        return std::move(frame);
    }

    RGBDFrame SaveFrame::frameLoadAuto(int frameId){

        RGBDFrame frame;
        frame.frameId = frameId;
        cv::Mat rgbBig = cv::imread(rgbPath + std::to_string(frame.frameId) + ".png",cv::IMREAD_COLOR);

        if(rgbBig.rows == 0){
            frame.frameId = -1;
            return frame;
        }

        cv::resize(rgbBig, frame.imRGB, cv::Size(640,480));
        cv::Mat depth255 = cv::imread(depthPath + std::to_string(frame.frameId) + ".png",-1);
        std::cout << depth255.rows << " depth "<<depth255.cols <<std::endl;

        depth255.convertTo(frame.imDepth, CV_32FC1);

        float tcwArr[4][4];

        std::ifstream tcwFile;
        tcwFile.open(tcwPath + std::to_string(frame.frameId) + ".txt");
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 4; ++k) {
                tcwFile >> tcwArr[i][k];
            }
        }

        cv::Mat tcw(4, 4, CV_32FC1, tcwArr);

        frame.mTcw = tcw.inv();

        return std::move(frame);

    }


}

