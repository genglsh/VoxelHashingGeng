#include <../include/SlamBase.h>
#include<chrono>

using namespace std::chrono;

namespace ark{
    orbAlignment::orbAlignment() {
        this->RT = cv::Mat::eye(4, 4, CV_64F);
        this->lastRT = cv::Mat::eye(4, 4, CV_64F);
        this->ID = -1;
        detector = cv::FeatureDetector::create("ORB");
        descriptor = cv::DescriptorExtractor::create("ORB");
    }

    bool orbAlignment::setCurrentFrame(const RGBDFrame& crtFrame,int FrameID) {
        this->ID = FrameID;
        if(FrameID) {
            this->lastFrame.imDepth = this->currentFrame.imDepth.clone();
            this->lastFrame.imRGB = this->currentFrame.imRGB.clone();
            this->lastFrame.frameId = this->currentFrame.frameId;
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
        // 声明特征提取器与描述子提取器
        auto sTime = system_clock::now();
//        cout << "align时间为　" << (float)alignT * microseconds::period::num / microseconds::period::den << "s"<< endl;
        vector< cv::KeyPoint > kp1, kp2; //关键点
//        detector->detect( (this->lastFrame).imRGB, kp1 );  //提取关键点
//        detector->detect( (this->currentFrame).imRGB, kp2 );

        // 计算描述子
        cv::Mat desp1, desp2;
//        descriptor->compute( (this->lastFrame).imRGB, kp1, desp1 );
//        descriptor->compute( (this->currentFrame).imRGB, kp2, desp2 );
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
        vector< cv::DMatch > matches;
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.match( desp1, desp2, matches );
//      cout<<"Find total "<<matches.size()<<" matches."<<endl;

        vector< cv::DMatch > goodMatches;
//        double minDis = 15;

//        double minDis = 20;
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
        for ( size_t i=0; i<matches.size(); i++ )
        {
//            if (matches[i].distance < 10*minDis)
//                goodMatches.push_back( matches[i] );
            if (matches[i].distance < minDist)
                goodMatches.push_back( matches[i] );
        }

        // 显示 good matches
//        cv::Mat imgMatches;
//        cout<<"good matches="<<goodMatches.size()<<endl;
//        cv::drawMatches( this->lastFrame.imRGB, kp1, this->currentFrame.imRGB, kp2, goodMatches, imgMatches );
//        cv::imshow( "good matches", imgMatches );
////        cv::imwrite( "./data/good_matches.png", imgMatches );
//        cv::waitKey(0);

        vector<cv::Point3f> pts_obj;
        // 第二个帧的图像点
        vector< cv::Point2f> pts_img;

        // 相机内参
        CAMERA_INTRINSIC_PARAMETERS C;
        C.cx = 305.432;
        C.cy = 250.41;
        C.fx = 517.448;
        C.fy = 517.448;
        C.scale = 1.0;

        for (size_t i=0; i<goodMatches.size(); i++)
        {
            cv::Point2f &p = kp1[goodMatches[i].queryIdx].pt;
//          cout << int(p.x) << " " << int(p.y) << endl;
            // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
            float &d = (this->currentFrame).imDepth.ptr<float>( int(p.y) )[ int(p.x) ];
            if (d == 0)
                continue;
            cv::Point2f &pImg = kp2[goodMatches[i].trainIdx].pt;
            pts_img.push_back(pImg);

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
        //原本最大次数为50次。
        cv::solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 50, 1.0, 100, inliers );
        // 求解pnp
//        cv::solvePnP( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false);

        // inliers 表示
        cout<<"inliers: "<<inliers.rows<<endl;
//        cout<<"R="<<rvec<<endl;
//        cout<<"t="<<tvec<<endl;

        cv::Rodrigues(rvec, R);
//        cout << R << endl;
        cv::Mat adjacentRT = cv::Mat::eye(4, 4, CV_64F);
        for (int i = 0; i < 3; i++)
            for(int ii = 0; ii < 3; ii++)
                adjacentRT.at<double>(i, ii) = R.at<double>(i,ii);

        for(int i = 0; i < 3; i++)
           adjacentRT.at<double>(i, 3) = tvec.at<double>(0, i);

//        adjacentRT = adjacentRT.inv();

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