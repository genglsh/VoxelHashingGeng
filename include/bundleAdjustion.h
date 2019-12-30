//
// Created by gengshuai on 19-12-13.
//

#ifndef TSDF_BUNDLEADJUSTION_H
#define TSDF_BUNDLEADJUSTION_H

#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

namespace ark {

    void saveSe3d(string fName, const Sophus::SE3d & tem);

    void find_feature_matches(
            const Mat &img_1, const Mat &img_2,
            std::vector<KeyPoint> &keypoints_1,
            std::vector<KeyPoint> &keypoints_2,
            std::vector<DMatch> &matches);

    // 像素坐标转相机归一化坐标
    Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
    typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
    typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

    void bundleAdjustmentG2O(
            const VecVector3d &points_3d,
            const VecVector2d &points_2d,
            const Mat &K,
            Sophus::SE3d &pose
    );

// BA by gauss-newton
    void bundleAdjustmentGaussNewton(
            const VecVector3d &points_3d,
            const VecVector2d &points_2d,
            const Mat &K,
            Sophus::SE3d &pose
    );

}

#endif //TSDF_BUNDLEADJUSTION_H
