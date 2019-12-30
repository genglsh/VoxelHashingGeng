#include <../include/SlamBase.h>
#include<chrono>
#include<bundleAdjustion.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;
using namespace std::chrono;

//#define _ORB

namespace ark{

    void convert_pts_to_matrix(const std::vector<cv::Point3f> &pts, cv::Mat &matrix) {
        matrix = cv::Mat(3, pts.size(), CV_32F);
        for (int i = 0; i < pts.size(); i++) {
            matrix.at<float>(0, i) = pts[i].x;
            matrix.at<float>(1, i) = pts[i].y;
            matrix.at<float>(2, i) = pts[i].z;
        }

    }

    orbAlignment::orbAlignment() {
        this->RT = cv::Mat::eye(4, 4, CV_64F);
        this->lastRT = cv::Mat::eye(4, 4, CV_64F);
        this->ID = -1;
//        detector = cv::FeatureDetector::create("ORB");
        detector = cv::ORB::create();
//        descriptor = cv::DescriptorExtractor::create("ORB");
        descriptor = cv::ORB::create();
//        cv::DenseFeatureDetector::create("SIFT");
//        detector = cv::Feature2D::create("SIFT");
    }

    bool estimate_RT(const std::vector<cv::Point3f> &srcPoints, const std::vector<cv::Point3f> &dstPoints,
                                 cv::Mat &R, cv::Mat &T) {
        if (srcPoints.size() != dstPoints.size() || srcPoints.size() < 3 || dstPoints.size() < 3) {
            std::cout << "srcPoints.size():\t" << srcPoints.size();
            std::cout << "dstPoints.size():\t" << dstPoints.size();
            std::cout << "registrateNPoint points size donot match!";
            return false;

        }
        float srcSumX = 0.0f;
        float srcSumY = 0.0f;
        float srcSumZ = 0.0f;

        float dstSumX = 0.0f;
        float dstSumY = 0.0f;
        float dstSumZ = 0.0f;

        size_t pointsNum = srcPoints.size();
        for (size_t i = 0; i < pointsNum; i++) {
            srcSumX += srcPoints[i].x;
            srcSumY += srcPoints[i].y;
            srcSumZ += srcPoints[i].z;

            dstSumX += dstPoints[i].x;
            dstSumY += dstPoints[i].y;
            dstSumZ += dstPoints[i].z;
        }
        cv::Point3f srcCentricPt(srcSumX / pointsNum, srcSumY / pointsNum, srcSumZ / pointsNum);
        cv::Point3f dstCentricPt(dstSumX / pointsNum, dstSumY / pointsNum, dstSumZ / pointsNum);
        cv::Mat srcMat;
        srcMat = cv::Mat::zeros(3, pointsNum, CV_32F);
        cv::Mat dstMat;
        dstMat = cv::Mat::zeros(3, pointsNum, CV_32F);
        for (size_t i = 0; i < pointsNum; ++i) {

            srcMat.at<float>(0, i) = srcPoints[i].x - srcCentricPt.x;
            srcMat.at<float>(1, i) = srcPoints[i].y - srcCentricPt.y;
            srcMat.at<float>(2, i) = srcPoints[i].z - srcCentricPt.z;

            dstMat.at<float>(0, i) = dstPoints[i].x - dstCentricPt.x;
            dstMat.at<float>(1, i) = dstPoints[i].y - dstCentricPt.y;
            dstMat.at<float>(2, i) = dstPoints[i].z - dstCentricPt.z;
        }

        cv::Mat matS = srcMat * dstMat.t();

        cv::Mat matU, matW, matV;
        cv::SVDecomp(matS, matW, matU, matV);

        cv::Mat matTemp = matU * matV;
        float det = cv::determinant(matTemp);

        float datM[] = {1, 0, 0, 0, 1, 0, 0, 0, det};
        cv::Mat matM(3, 3, CV_32FC1, datM);

        cv::Mat matR = matV.t() * matM * matU.t();
        float tx, ty, tz;
        tx = dstCentricPt.x - (srcCentricPt.x * matR.at<float>(0, 0) + srcCentricPt.y * matR.at<float>(0, 1) +
                               srcCentricPt.z * matR.at<float>(0, 2));
        ty = dstCentricPt.y - (srcCentricPt.x * matR.at<float>(1, 0) + srcCentricPt.y * matR.at<float>(1, 1) +
                               srcCentricPt.z * matR.at<float>(1, 2));
        tz = dstCentricPt.z - (srcCentricPt.x * matR.at<float>(2, 0) + srcCentricPt.y * matR.at<float>(2, 1) +
                               srcCentricPt.z * matR.at<float>(2, 2));
        float datT[] = {tx, ty, tz};
        cv::Mat matT(3, 1, CV_32F, datT);
        matR.copyTo(R);
        matT.copyTo(T);
        return true;
    }

    bool estimate_RT_ransac(const std::vector<cv::Point3f> &srcPoints, const std::vector<cv::Point3f> &dstPoints,
                                   cv::Mat &R, cv::Mat &T, std::vector<int> &inliers, const int mini_matches,
                                   const float matched_min_d, const float accept_ratio, const int max_iters) {

        srand(0);

        //check the legality
        if (srcPoints.size() != dstPoints.size()) {
            printf("\nThe number of source points must be equal to the dst points");
            return false;
        }
        if (srcPoints.size() < 3) {
            printf("\nToo few points for estimation(must be >=3");
            return false;
        }
        if (mini_matches > srcPoints.size()) {
            printf("\nThere are not enough points for esimation. Requried is: %d while only provided:%d", mini_matches,
                   srcPoints.size());
            return false;
        }

        float max_ratio = 0;
        Mat bst_R, bst_T;


        for (int i = 0; i < max_iters; i++) {

            //select the matches for RT estiamtion
            vector<Point3f> selected_src_points;
            vector<Point3f> selected_dst_points;
            vector<int> shuffled_index;
            for (int i = 0; i < srcPoints.size(); i++) {
                shuffled_index.push_back(i);
            }
            random_shuffle(shuffled_index.begin(), shuffled_index.end());
            for (int i = 0; i < mini_matches; i++) {
                selected_src_points.push_back(srcPoints[shuffled_index[i]]);
                selected_dst_points.push_back(dstPoints[shuffled_index[i]]);
            }
            //get the initial RT
            Mat tempR, tempT;
            bool suc = estimate_RT(selected_src_points, selected_dst_points, tempR, tempT);

//            cout << tempR << tempT << endl;
            if (suc == false) {
                printf("\nRT estimation fails");
                return false;
            }
            //check the satisfied points
            Mat src_pts_matrix, dst_pts_matrix, predicted_dst_pts_matrix;
            convert_pts_to_matrix(srcPoints, src_pts_matrix);
            convert_pts_to_matrix(dstPoints, dst_pts_matrix);

            predicted_dst_pts_matrix = tempR * src_pts_matrix;
            float satisfied_count = 0;
            Mat diff;
            inliers.clear();
            for (int i = 0; i < srcPoints.size(); i++) {
                Mat predicted_dst_pt = predicted_dst_pts_matrix.col(i);
                predicted_dst_pt = predicted_dst_pt + tempT;
                diff = predicted_dst_pt - dst_pts_matrix.col(i);
                Mat inner_product = diff.t() * diff;
                float dist = sqrt(inner_product.at<float>(0,0));
//                cout << "dist is " << dist << endl;
                if (dist < matched_min_d) {
                    satisfied_count++;
                    inliers.push_back(i);
//                cout<<endl<<" "<<predicted_dst_pt << " "<<dst_pts_matrix.col(i)<< "with dist:"<< dist;
//                printf("\ndist:%f", dist);
                }
            }
            float passed_ratio = satisfied_count / srcPoints.size();
//            cout << "passed_ratio " << passed_ratio << endl;
            if(passed_ratio > max_ratio){
                bst_R = tempR;
                bst_T = tempT;
                max_ratio = passed_ratio;
            }
            vector<Point3f> src_pts_final, dst_pts_final;
            //if find the correct rotation and translation
            if (passed_ratio > accept_ratio) {
                for (int i = 0; i < inliers.size(); i++) {
                    src_pts_final.push_back(srcPoints[inliers[i]]);
                    dst_pts_final.push_back(dstPoints[inliers[i]]);
                }
                bool suc = estimate_RT(src_pts_final, dst_pts_final, R, T);
                if(suc == false){
                    printf("\nRANSAC failure when using more consensussed matches. Might be due to numerical reasons!");
                    return false;
                }

                printf("\nIterated number for RANSAC :%d with %d pairs of points", i , inliers.size());
                return true;

            }
        }
        //If not satisfied in the loop, then return the best R and T.
        printf("\nNot able to find the satisfied transforamtion, return the best in the iterations");
        R = bst_R;
        T = bst_T;
        return false;


    }


    bool orbAlignment::setCurrentFrame(const RGBDFrame& crtFrame,int FrameID) {
        this->ID = FrameID;
        if(FrameID) {
            this->lastFrame.imDepth = this->currentFrame.imDepth.clone();
            this->lastFrame.imRGB = this->currentFrame.imRGB.clone();
            this->lastFrame.frameId = this->currentFrame.frameId;
            this->lastFrame.mTcw = this->currentFrame.mTcw;
        }

        this->currentFrame =  crtFrame;

    }

    cv::Point3f orbAlignment::point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera ) {
        cv::Point3f p;
        p.z = double(point.z);
        p.x = ( point.x - camera.cx) * p.z / camera.fx;
        p.y = ( point.y - camera.cy) * p.z / camera.fy;
        return std::move(p);
    }

    void orbAlignment::ComputationalCharacteristics(vector<cv::KeyPoint>& kp, cv::Mat& des, const cv::Mat& rgb){
        detector->detect( rgb, kp);
        descriptor->compute(rgb, kp, des);
    }

    
    bool orbAlignment::align() {
        auto sTime = system_clock::now();
//        cout << "align时间为　" << (float)alignT * microseconds::period::num / microseconds::period::den << "s"<< endl;
        vector< cv::KeyPoint > kp1, kp2; //关键点
        vector< cv::DMatch > matches;
        vector< cv::DMatch > goodMatches;
#ifdef _ORB

        cv::Mat desp1, desp2;

        std::thread lastFeature(&orbAlignment::ComputationalCharacteristics,this,
                std::ref(kp1), std::ref(desp1), std::ref(currentFrame.imRGB)); // 创建生产者线程.
        std::thread currentFeature(&orbAlignment::ComputationalCharacteristics,this,
                                   std::ref(kp2), std::ref(desp2), std::ref(lastFrame.imRGB));
        lastFeature.join();
        currentFeature.join();
        auto eTime = system_clock::now();
        auto featureT = duration_cast<std::chrono::microseconds>(eTime - sTime).count();
        cout << "feature时间为　" << (float)featureT * microseconds::period::num / microseconds::period::den << "s"<< endl;

        // 匹配描述子

        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        matcher.match( desp1, desp2, matches );
//      cout<<"Find total "<<matches.size()<<" matches."<<endl;


        double minDist = 30;
//        printf("minDis is %lf\n", minDis);
        //之前策略遍历寻找最小距离，现在发现经常会被设置为30，所以直接默认设置为30；
//        for( int i = 0; i < matches.size(); i++) {
//            double dist = matches[i].distance;
//            if(dist < minDist)
//                minDist = dist;
//        }
//
//        if(minDist < 30){
//            minDist = 30;
//        }
        cout << "minDis is " << minDist<<endl;


        // 最小距离方法为选取最近匹配距离
        for ( size_t i=0; i<matches.size(); i++ ) {
//            if (matches[i].distance < 10*minDis)
//                goodMatches.push_back( matches[i] );
            if (matches[i].distance < minDist)
                goodMatches.push_back( matches[i] );
        }
#else

        Mat descriptors_1, descriptors_2;
        // used in OpenCV3
        // Ptr<FeatureDetector> detector = cv::FeatureDetector::create("ORB");
        // Ptr<DescriptorExtractor> descriptor = cv::DescriptorExtractor::create("ORB");
        cv::Ptr<cv::xfeatures2d::SIFT> detectorS = cv::xfeatures2d::SIFT::create();
        // use this if you are in OpenCV2
        // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
        // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");

        //-- 第一步:检测 Oriented FAST 角点位置
        // detector->detect(img_1, keypoints_1);
        // detector->detect(img_2, keypoints_2);

        // //-- 第二步:根据角点位置计算 BRIEF 描述子
        // descriptor->compute(img_1, keypoints_1, descriptors_1);
        // descriptor->compute(img_2, keypoints_2, descriptors_2);


        detectorS->detectAndCompute(this->currentFrame.imRGB, cv::Mat(), kp1, descriptors_1);
        detectorS->detectAndCompute(this->lastFrame.imRGB, cv::Mat(), kp2, descriptors_2);
        //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
        vector<DMatch> match;
        // BFMatcher matcher ( NORM_HAMMING );
        matcher->match(descriptors_1, descriptors_2, match);
        cout << descriptors_1.rows << " " << descriptors_1.cols << endl;

        //-- 第四步:匹配点对筛选
        double minDist = 50, max_dist = 0, min_dist = 999;

        //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
        for (int i = 0; i < descriptors_1.rows; i++) {
            double dist = match[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        printf("-- Max dist : %f \n", max_dist);
        printf("-- Min dist : %f \n", min_dist);

        //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
        for (int i = 0; i < descriptors_1.rows; i++) {

//            cout << "distance is " << match[i].distance << endl;
            if (match[i].distance <= max_dist* 0.5) {
                goodMatches.push_back(match[i]);
            }
        }

#endif
        // 显示 good matches
//        cv::Mat imgMatches;
//        cout<<"good matches="<<goodMatches.size()<<endl;
//        cv::drawMatches( this->currentFrame.imRGB, kp1, this->lastFrame.imRGB, kp2, goodMatches, imgMatches );
//        cv::imshow( "good matches", imgMatches );
////        cv::imwrite( "./data/good_matches.png", imgMatches );
//        cv::waitKey(0);

        vector<cv::Point3f> pts_obj;
        // 第二个帧的图像点
        vector< cv::Point2f> pts_img;
        vector< cv::Point3f> pts_target;
        cout << "goodMatch size is " << goodMatches.size() << endl;

        // 相机内参
        CAMERA_INTRINSIC_PARAMETERS C;
        C.cx = 305.432;
        C.cy = 250.41;
        C.fx = 517.448;
        C.fy = 517.448;
        C.scale = 1.0;

        for (size_t i=0; i < goodMatches.size(); i++) {

            cv::Point2f &p = kp1[goodMatches[i].queryIdx].pt;
            cv::Point2f &pImg = kp2[goodMatches[i].trainIdx].pt;
//          cout << int(p.x) << " " << int(p.y) << endl;
            // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
            float &d = (this->currentFrame).imDepth.ptr<float>( int(p.y) )[ int(p.x) ];

            float &dtarget = (this->lastFrame).imDepth.ptr<float>( int(pImg.y) )[ int(pImg.x) ];
            if (d == 0 || dtarget == 0)
                continue;

            pts_img.push_back(pImg);
            cv::Point3f pttarget (pImg.x, pImg.y, dtarget);
            cv::Point3f pdtarget  = point2dTo3d(pttarget, C );
            pts_target.push_back(pdtarget);

            // 将(u,v,d)转成(x,y,z)
            cv::Point3f pt (p.x, p.y, d);
            cv::Point3f pd = point2dTo3d(pt, C );
            pts_obj.push_back(pd);

        }
//    cout << pts_img.size() << endl;

        double camera_matrix_data[3][3] = {
                {C.fx, 0, C.cx},
                {0, C.fy, C.cy},
                {0, 0, 1}
        };

        // 构建相机矩阵
        cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data);
        cv::Mat rvec, tvec, inliers, R;
        vector<int> inlierVec;
        ark::estimate_RT_ransac(pts_obj, pts_target,rvec, tvec, inlierVec, 10, 3, 0.9, 300);

        cout << rvec << endl;
        cout << tvec << endl;

        rvec.convertTo(rvec, CV_64F);
        tvec.convertTo(tvec, CV_64F);
//        cout <<  << endl;
//        cout << tvec.type << endl;
        //原本最大次数为50次。
//        cv::solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, true, 100, 1.0, 0.99, inliers);
//        cv::solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 300, 1, 100, inliers );
        // 求解pnp
//        cv::solvePnP( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false);

        // inliers 表示
        cout<<"inliers: "<<inliers.rows<<endl;
//        cout<<"R="<<rvec<<endl;
//        cout<<"t="<<tvec<<endl;

//        cv::Rodrigues(rvec, R);
//        cout << R << endl;
        cv::Mat adjacentRT = cv::Mat::eye(4, 4, CV_64F);
        for (int i = 0; i < 3; i++)
            for(int ii = 0; ii < 3; ii++)
                adjacentRT.at<double>(i, ii) = rvec.at<double>(i,ii);

        for(int i = 0; i < 3; i++)
           adjacentRT.at<double>(i, 3) = tvec.at<double>(0, i);

//        adjacentRT = adjacentRT.inv();

        cout << "adjacentRT" << adjacentRT << endl;

        VecVector3d pts_3d_eigen;
        VecVector2d pts_2d_eigen;
         for (size_t i = 0; i < pts_obj.size(); ++i) {
             pts_3d_eigen.push_back(Eigen::Vector3d(pts_obj[i].x, pts_obj[i].y, pts_obj[i].z));
             pts_2d_eigen.push_back(Eigen::Vector2d(pts_img[i].x, pts_img[i].y));
         }

         cout << "calling bundle adjustment by gauss newton" << endl;
//         Sophus::SE3d pose_gn;
         cout << "adjacentRt" << adjacentRT << endl;
//         bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, cameraMatrix, pose_gn);

//        Sophus::SE3d pose_g2o;
//         t1 = chrono::steady_clock::now();
//         bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, cameraMatrix, pose_g2o);

        this->RT = this->lastRT * adjacentRT;
//        cout << "RT is\n" << this->RT <<endl;
        this->lastRT = this->RT;
    }
}

//int main( int argc, char** argv )
//{
//    cv::Mat R;
//    cv::Mat lastR = cv::Mat::eye(4, 4, CV_64F);
//    int sampleNum = 100;
//    for (int x = 0; x < sampleNum; x++) {
//        // 声明并从data文件夹里读取两个rgb与深度图
//        cv::Mat this->lastFrame.imRGB = cv::imread( "../scene0220_02/RGB/" + std::to_string(x)+".png");
//        cv::Mat this->currentFrame.imRGB = cv::imread( "../scene0220_02/RGB/" + std::to_string(x+1)+".png");
//        cv::Mat this->lastFrame.imDepth = cv::imread( "../scene0220_02/depth/" + std::to_string(x) + ".png", -1);
//        cv::Mat this->currentFrame.imDepth = cv::imread( "../scene0220_02/depth/" + std::to_string(x+1) + ".png", -1);
//        // 声明特征提取器与描述子提取器
//        cv::Ptr<cv::FeatureDetector> detector;
//        cv::Ptr<cv::DescriptorExtractor> descriptor;
//
//        detector = cv::FeatureDetector::create("ORB");
//        descriptor = cv::DescriptorExtractor::create("ORB");
//
//        vector< cv::KeyPoint > kp1, kp2; //关键点
//        detector->detect( this->lastFrame.imRGB, kp1 );  //提取关键点
//        detector->detect( this->currentFrame.imRGB, kp2 );
//
////        cout<<"Key points of two images: "<<kp1.size()<<", "<<kp2.size()<<endl;
//
//        // 可视化， 显示关键点
//        cv::Mat imgShow;
//        cv::drawKeypoints( this->lastFrame.imRGB, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
//        cv::imshow( "keypoints", imgShow );
////  cv::imwrite( "./data/keypoints.png", imgShow );
//        cv::waitKey(0); //暂停等待一个按键
//
//        // 计算描述子
//        cv::Mat desp1, desp2;
//        descriptor->compute( this->lastFrame.imRGB, kp1, desp1 );
//        descriptor->compute( this->currentFrame.imRGB, kp2, desp2 );
//
//        // 匹配描述子
//        vector< cv::DMatch > matches;
//        cv::BFMatcher matcher;
//        matcher.match( desp1, desp2, matches );
////      cout<<"Find total "<<matches.size()<<" matches."<<endl;
//
//        // 可视化：显示匹配的特征
//        cv::Mat imgMatches;
//        cv::drawMatches( this->lastFrame.imRGB, kp1, this->currentFrame.imRGB, kp2, matches, imgMatches );
//        cv::imshow( "matches", imgMatches );
////      cv::imwrite( "./data/matches.png", imgMatches );
//        cv::waitKey( 0 );
//
//        // 筛选匹配，把距离太大的去掉
//        // 这里使用的准则是去掉大于四倍最小距离的匹配
//        vector< cv::DMatch > goodMatches;
//        double minDis = 15;
//        // 最小距离方法为选取最近匹配距离
////        for ( size_t i=0; i<matches.size(); i++ )
////        {
////            if ( matches[i].distance < minDis )
////                minDis = matches[i].distance;
////        }
////        if (minDis == 0) {
////            minDis += 1;
////        }
//        cout<<"min dis = "<<minDis<<endl;
//
//        for ( size_t i=0; i<matches.size(); i++ )
//        {
//            if (matches[i].distance < 10*minDis)
//                goodMatches.push_back( matches[i] );
//        }
//
//        // 显示 good matches
//        cout<<"good matches="<<goodMatches.size()<<endl;
//        cv::drawMatches( this->lastFrame.imRGB, kp1, this->currentFrame.imRGB, kp2, goodMatches, imgMatches );
//        cv::imshow( "good matches", imgMatches );
////        cv::imwrite( "./data/good_matches.png", imgMatches );
//        cv::waitKey(0);
//
//        // 计算图像间的运动关系
//        // 关键函数：cv::solvePnPRansac()
//        // 为调用此函数准备必要的参数
//
//        // 第一个帧的三维点
//        vector<cv::Point3f> pts_obj;
//        // 第二个帧的图像点
//        vector< cv::Point2f > pts_img;
//
//        // 相机内参
//        CAMERA_INTRINSIC_PARAMETERS C;
//        C.cx = 305.432;
//        C.cy = 250.41;
//        C.fx = 517.448;
//        C.fx = 517.448;
//        C.fy = 517.448;
//        C.scale = 1000.0;
//
//        for (size_t i=0; i<goodMatches.size(); i++)
//        {
//            // query 是第一个, train 是第二个
//            cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;
////          cout << int(p.x) << " " << int(p.y) << endl;
//            // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
//            ushort d = this->lastFrame.imDepth.ptr<ushort>( int(p.y) )[ int(p.x) ];
//            if (d == 0)
//                continue;
//            pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );
//
//            // 将(u,v,d)转成(x,y,z)
//            cv::Point3f pt (p.x, p.y, d);
//            cv::Point3f pd = point2dTo3d(pt, C );
//            pts_obj.push_back( pd );
//        }
////    cout << pts_img.size() << endl;
//
//        double camera_matrix_data[3][3] = {
//                {C.fx, 0, C.cx},
//                {0, C.fy, C.cy},
//                {0, 0, 1}
//        };
//
//        // 构建相机矩阵
//        cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
//        cv::Mat rvec, tvec, inliers;
//        // 求解pnp
//        cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );
//
//        cout<<"inliers: "<<inliers.rows<<endl;
//        cout<<"R="<<rvec<<endl;
//        cout<<"t="<<tvec<<endl;
//
//        cv::Rodrigues(rvec, R);
//        cout << R << endl;
//        cv::Mat RT = cv::Mat::eye(4, 4, CV_64F);
//        for (int i = 0; i < 3; i++)
//            for(int ii = 0; ii < 3; ii++)
//                RT.at<double>(i, ii) = R.at<double>(i,ii);
//
//        for(int i = 0; i < 3; i++)
//            RT.at<double>(i, 3) = tvec.at<double>(0, i);
//
//        RT = lastR * RT;
//        cout << "RT is\n" << RT <<endl;
//        string fileName = "../scene0220_02/tcw/" + std::to_string(x+1) + ".txt";
//        cout << "fileName is \n" << fileName;
//        std::ofstream openfile(fileName);
//        for (int i = 0; i < 4; i++) {
//            for (int ii = 0; ii < 4; ii++) {
//                if(ii < 3)
//                    openfile << RT.at<double>(i,ii) << " ";
//                else
//                    openfile << RT.at<double>(i,ii) << "\n";
//            }
//        }
//        openfile.close();
//        lastR = RT;
//    }
//    return 0;
//}