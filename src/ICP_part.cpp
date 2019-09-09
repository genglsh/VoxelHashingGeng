#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <string>
#include <sstream>
using namespace std;
using pcl::PointXYZ;
using pcl::PointCloud;

Eigen::Matrix4f get_align_matrix(PointCloud<PointXYZ>::Ptr cloud_source, PointCloud<PointXYZ>::Ptr cloud_target,
                                 Eigen::Matrix4f init_mat){
    pcl::IterativeClosestPoint<PointXYZ, PointXYZ> icp;
    cloud_source->is_dense = false;
    cloud_target->is_dense = false;
    PointCloud<PointXYZ>::Ptr cloud_source_registered(new PointCloud<PointXYZ>());
    icp.setInputSource(cloud_source);
    icp.setInputTarget (cloud_target);
    icp.setMaxCorrespondenceDistance (10);
    icp.setMaximumIterations (50);
    icp.setTransformationEpsilon (1e-12);
    icp.setEuclideanFitnessEpsilon (1e-7);

    icp.align(*(cloud_source_registered), init_mat);

    Eigen::Matrix4f transformation = icp.getFinalTransformation ();
    cout<<"最优对齐结果"<<icp.getFitnessScore()<<endl;
    return transformation;
}

void open_Point_XYZ(char* name, PointCloud<PointXYZ>::Ptr pc){
    ifstream fin(name);
    if (fin.fail())
    {
        cout<<"打开文件错误!"<<endl;
    }
    string tem;
    while(getline(fin,tem)){
        stringstream tem_buffer(tem);
        float a, b, c;
        tem_buffer >> a;
        tem_buffer >> b;
        tem_buffer >> c;
        pc->push_back(*(new PointXYZ(a, b, c)));
    }
    fin.close();
}

void open_eigen_mat(char* name, Eigen::Matrix4f& init){
    ifstream fin(name);
    if (fin.fail())
    {
        cout<<"打开文件错误!"<<endl;
    }
    string tem;
    int cnt = 0;
    while(getline(fin, tem)) {
        stringstream tem_buffer(tem);
        float a, b, c, d;
        tem_buffer >> a;
        tem_buffer >> b;
        tem_buffer >> c;
        tem_buffer >> d;
        init(cnt, 0) = a;
        init(cnt, 1) = b;
        init(cnt, 2) = c;
        init(cnt, 3) = d;
        cnt++;
    }
}

int main(int argv, char** argc){
    PointCloud<PointXYZ>::Ptr pc1(new PointCloud<PointXYZ>());
    PointCloud<PointXYZ>::Ptr pc2(new PointCloud<PointXYZ>());
    open_Point_XYZ("/home/gengshuai/Downloads/PicoZenseSDK_Ubuntu16.04_20190316_v2.3.9.2_DCAM710/Samples/FrameViewer/PointCloud0.txt",pc1);
    open_Point_XYZ("/home/gengshuai/Downloads/PicoZenseSDK_Ubuntu16.04_20190316_v2.3.9.2_DCAM710/Samples/FrameViewer/PointCloud1.txt", pc2);
    Eigen::Matrix4f init = Eigen::Matrix4f::Identity();
    open_eigen_mat("/home/gengshuai/Desktop/graduate/test/Voxel-Hashing-SDF/scene0220_02/tcw/0.txt",init);
    cout << "init_transform" << init << endl;
    Eigen::Matrix4f tran =  get_align_matrix(pc2,pc1, init);
    cout << tran << endl;
}

