//
// Created by gengshuai on 19-11-3.
//
#include "ObjectExtract.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
using namespace cv;

namespace ark {
    const int maxDepth = 3000;
//    int OtsuAlgThreshold(const Mat& image) {
//        if (image.channels() != 1) {
//            cout << "Please input Gray-image!" << endl;
//            return 0;
//        }
//        int T = 0; //Otsu算法阈值
//        double varValue = 0; //类间方差中间值保存
//        double w0 = 0; //前景像素点数所占比例
//        double w1 = 0; //背景像素点数所占比例
//        double u0 = 0; //前景平均灰度
//        double u1 = 0; //背景平均灰度
//         //灰度直方图，下标是灰度值，保存内容是灰度值对应的像素点总数
//        float *data = image.data;
//        double totalNum = image.rows * image.cols; //像素总数
//        //计算灰度直方图分布，Histogram数组下标是灰度值，保存内容是灰度值对应像素点数
//        for (int i = 0; i < image.rows; i++)   //为表述清晰，并没有把rows和cols单独提出来
//        {
//            for (int j = 0; j < image.cols; j++) {
//                Histogram[data[i * image.step + j]]++;
//            }
//        }
//        for (int i = 0; i < 3000; i++) {
//            //每次遍历之前初始化各变量
//            w1 = 0;
//            u1 = 0;
//            w0 = 0;
//            u0 = 0;
//            //***********背景各分量值计算**************************
//            for (int j = 0; j <= i; j++) //背景部分各值计算
//            {
//                w1 += Histogram[j];  //背景部分像素点总数
//                u1 += j * Histogram[j]; //背景部分像素总灰度和
//            }
//            if (w1 == 0) //背景部分像素点数为0时退出
//            {
//                continue;
//            }
//            u1 = u1 / w1; //背景像素平均灰度
//            w1 = w1 / totalNum; // 背景部分像素点数所占比例
//            //***********背景各分量值计算**************************
//
//            //***********前景各分量值计算**************************
//            for (int k = i + 1; k < 3000; k++) {
//                w0 += Histogram[k];  //前景部分像素点总数
//                u0 += k * Histogram[k]; //前景部分像素总灰度和
//            }
//            if (w0 == 0) //前景部分像素点数为0时退出
//            {
//                break;
//            }
//            u0 = u0 / w0; //前景像素平均灰度
//            w0 = w0 / totalNum; // 前景部分像素点数所占比例
//            //***********前景各分量值计算**************************
//
//            //***********类间方差计算******************************
//            double varValueI = w0 * w1 * (u1 - u0) * (u1 - u0); //当前类间方差计算
//            if (varValue < varValueI) {
//                varValue = varValueI;
//                T = i;
//            }
//        }
//        return T;
//
//    }

    int globalThresholdPM(Mat& image) {
        float *data = image.data;
        int Histogram[maxDepth] = {0};
        int num[maxDepth] = {0};
        int depthValue[maxDepth] = {0};

        int T = 1000; //初始化阈值。
        int numPoints = image.rows*image.cols;
        for(int x = 0; x < numPoints, x++) {
            if( data[x] < maxDepth)
                Histogram[data[x]]++;
        }

        uint64_t depthV = 0;
        uint32_t countN = 0;
        for(int x = 0; x < maxDepth; x++) {
            depthV += Histogram[x] * x;
            countN += Histogram[x];
            depthValue[x] = depthV;
            num[x] = countN;
        }
        float delT = 1000; //设置初始较大delta
        float delTThreshold = 10;
        while (delT < delTThreshold){

            float leftPart = (depthValue[T]*1.0 / num[T]);
            float rightPart = ((depthValue[maxDepth-1]-depthValue[T])*1.0 / (num[maxDepth-1] - num[T]));
            int NewT = (leftPart + rightPart) / 2.0;
            delT = abs(NewT - T);
            T = NewT;
        }
        return T;
    }
}


