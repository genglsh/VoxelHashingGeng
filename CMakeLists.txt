cmake_minimum_required(VERSION 2.8)
project(TSDF)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")
set( CMAKE_CXX_FLAGS "-std=c++11" )

#set( DBoW3_INCLUDE_DIRS "/usr/local/include/" )
#set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a" )

#Eigen
SET( EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3")
include_directories(${EIGEN3_INCLUDE_DIR})

#set(G2O_DIR "/home/gengshuai/Downloads/g2oGithub/g2o/cmake_modules")
LIST( APPEND CMAKE_MODULE_PATH "/home/gengshuai/Downloads/g2oGithub/g2o/cmake_modules")
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O REQUIRED)
include_directories( ${G2O_INCLUDE_DIRS})

#CSparse

FIND_PACKAGE( CSparse REQUIRED )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )

#SET( Sophus_DIR "/usr/local/include/eigen3")
find_package( Sophus REQUIRED )
if (NOT Sophus_FOUND)
    MESSAGE(FATAL_ERROR "fail")
endif ()
include_directories( ${Sophus_INCLUDE_DIRS})


#SET( OpenCV_DIR "/home/gengshuai/Downloads/opencv-3.4.8")
#find_package(OpenCV 2.4.3 QUIET)

#find_package(OpenCV 3.4.8 REQUIRED PATHS "/home/Documents/env/opencv3")
#SET(OpenCV_DIR /home/Documents/env/opencv3)
#find_package(OpenCV )
find_package(OpenCV 4.2.0 REQUIRED PATHS "/media/gengshuai/8e859ff5-c0b1-42ae-bf0e-9102ed601b72/Documents/env/opencv330")
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV 3 not found.")
    find_package(OpenCV 2.4.3 QUIET)
    #    find_package(OpenCV 3.0 QUIET)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif ()
endif ()
#message("opencv is " Opencv_)


set(OB_DIRECTORY /home/gengshuai/Downloads/OpenNI2.3.0.61-Bate/2.3.0.61/Demo/UVCDemo/Linux/OrbbecStreamSample-Linux/OrbbecStreamSample-Linux/)

include_directories(
        ${OB_DIRECTORY}/include
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
        ${DBoW3_INCLUDE_DIRS}

)

find_package(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)
SET(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
cuda_add_library(${PROJECT_NAME} SHARED src/tsdf.cu include/tsdf.cuh)



MESSAGE(STATUS "opencv Version" ${OpenCV_VERSION})


find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

link_directories(/home/gengshuai/Downloads/OpenNI2.3.0.61-Bate/2.3.0.61/Demo/UVCDemo/Linux/OrbbecStreamSample-Linux/OrbbecStreamSample-Linux)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Example)

add_executable(load_frames src/main.cpp
        src/RgbPart.cpp
        src/PointCloudGenerator.cpp
        src/SaveFrame.cpp
        src/safecall.cpp
        src/Utils.cpp
        src/ridOfPlane.cpp)

target_link_libraries(load_frames
    ${PROJECT_NAME}
#        ${DBoW3_LIBS}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${Sophus_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension
    cxsparse
    cholmod
    g2o_cli g2o_ext_freeglut_minimal g2o_simulator
    g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
    g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
    g2o_types_sba g2o_types_slam3d g2o_csparse_extension
    g2o_opengl_helper g2o_solver_dense g2o_stuff
    g2o_types_sclam2d g2o_parser g2o_solver_pcg
    g2o_types_data g2o_types_sim3
    glog
    OpenNI2
	glut GL GLU GLEW)

