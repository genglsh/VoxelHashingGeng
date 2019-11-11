//save Main;
#include <iostream>
#include <algorithm>
#include <thread>
#include <queue>
#include <chrono>
#include <time.h>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <vector>


#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>


#include <GL/glew.h>
#include <GL/glut.h>

#include <opencv2/opencv.hpp>

#include <SaveFrame.h>
//#include <ORBSLAMSystem.h>
//#include <BridgeRSD435.h>
#include <PointCloudGenerator.h>
#include <opencv2/core/eigen.hpp>
#include <ICP_part.h>
#include "preview.h"
#include <SlamBase.h>
#include <Utils.h>
#include "ObjectExtract.h"

using namespace std;
using namespace std::chrono;

//OpenGL global variable
float window_width = 800;
float window_height = 800;
float xRot = 15.0f;
float yRot = 0.0f;
float xTrans = 0.0;
float yTrans = 0;
float zTrans = -35.0;
int ox;
int oy;
int buttonState;
float xRotLength = 0.0f;
float yRotLength = 0.0f;
bool wireframe = false;
bool stop = false;

ark::PointCloudGenerator *pointCloudGenerator;
ark::SaveFrame *saveFrame;
std::thread *app;
ark::ICPPart *ICP;
ark::orbAlignment *ORBAlignment;

using namespace std;

static const int kItemsToProduce  = 30;   // How many items we plan to produce.

struct ItemRepository {
    queue<ark::RGBDFrame> item_buffer; // 产品缓冲区, 配合 read_position 和 write_position 模型环形队列.
    std::mutex mtx; // 互斥量,保护产品缓冲区
    std::condition_variable repo_not_full; // 条件变量, 指示产品缓冲区不为满.
    std::condition_variable repo_not_empty; // 条件变量, 指示产品缓冲区不为空.
    std::condition_variable repoRT_not_empty;
    queue<cv::Mat> RTQueue;
    std::mutex mtxRT;

    ItemRepository(){
        RTQueue.push(cv::Mat::eye(4,4,CV_64F));
    }

} gItemRepository; // 产品库全局变量, 生产者和消费者操作该变量

typedef struct ItemRepository ItemRepository;

void AlignPart(ItemRepository* it){

    std::unique_lock<std::mutex> lock(it->mtxRT);
    ORBAlignment->align();
//    cv::Mat temRT = ORBAlignment->RT;
    it->RTQueue.push(ORBAlignment->RT);
    it->repoRT_not_empty.notify_all();
    lock.unlock();
}

cv::Mat GetRTPart(ItemRepository* it) {

    std::unique_lock<std::mutex> lock(it->mtxRT);
    if(it->RTQueue.empty()){
        it->repoRT_not_empty.wait(lock);
    }
    cv::Mat tem = it->RTQueue.front();
    it->RTQueue.pop();
    lock.unlock();
    return std::move(tem);
}

void FusionPart(ark::RGBDFrame& frame, int cnt){

    system_clock::time_point startTime = system_clock::now();
/*
 * 利用ICP方法来做对齐。
    bool KeyFrame = ICP->CVTimage2Point(cnt, frame.imDepth);
    if (!KeyFrame) {
        return;
    }

    Eigen::Matrix4f show = ICP->getRT();
    cout << ICP->frameID << endl;
    cout << ICP->currentFrame->size()<<endl;
    cout << ICP->lastFrame->size() << endl;
    cout << "current RT is \n" << show << endl;
//    Eigen::Matrix4f showInv = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f showInv = show.inverse();
    cv::eigen2cv(showInv, frame.mTcw);
    cout << frame.mTcw << endl;
    // 这个地方还需要注意视频流提供的彩色图是RGB还是BGR， 需不需要进行二次转换。
    cout << "frame data" << endl;
*/

//    利用orb方法来做对齐。

    bool setFlag = ORBAlignment->setCurrentFrame(frame, cnt);
    system_clock::time_point setFrameTime = system_clock::now();
    ark::printTime(startTime, setFrameTime, "set 时间为");
//    auto setTime = duration_cast<std::chrono::microseconds>(setFrameTime - startTime).count();
//    cout << "set时间为　" << (float)setTime * microseconds::period::num / microseconds::period::den << "s"<< endl;

//    ark::countTime(startTime, setFrameTime);
    if(cnt){
        bool alignFlag = ORBAlignment->align();
//        cout << "当前帧为"<< cnt << endl;
//        cout<< ORBAlignment->RT << endl;
    }
    auto alignTime = system_clock::now();
    auto alignT = duration_cast<std::chrono::microseconds>(alignTime - setFrameTime).count();
    cout << "align时间为　" << (float)alignT * microseconds::period::num / microseconds::period::den << "s"<< endl;

//    ark::countTime(setFrameTime, alignTime);

//    cout << "depth 阈值 is " << ark::globalThresholdPM(frame.imDepth)<< endl;
//    system_clock::time_point fbTime = system_clock::now();
    auto baTime = system_clock::now();
    pointCloudGenerator->SetMaxDepth(ark::globalThresholdPM(frame.imDepth));
    auto bafTime = system_clock::now();
    ark::printTime(baTime, bafTime, "前后景分割处理时间");
//    system_clock::time_point fbendTime = system_clock::now();
//    auto durationbf = duration_cast<std::chrono::microseconds>(fbendTime - fbTime).count();
//    cout << "前后景分割处理时间为　" << (float)durationbf * microseconds::period::num / microseconds::period::den << "s"<< endl;

    frame.mTcw = (ORBAlignment->RT);
    cout<< frame.mTcw;

//    cv::imshow("rgb",frame.imRGB);
//    cv::waitKey(0);
//    cv::imshow("depth", frame.imDepth);
//    cv::waitKey(0);
//    cout << frame.mTcw<<endl;

    cout<< frame.imRGB.size().width << " " << frame.imRGB.size().height << " " << frame.imRGB.channels()<<endl;


    auto cvtColorT = system_clock::now();

    cv::cvtColor(frame.imRGB, frame.imRGB, cv::COLOR_BGR2RGB);

    auto cvtColorA = system_clock::now();
    ark::printTime(cvtColorT, cvtColorA, "颜色转换时间");
//    cv::Mat Twc = frame.mTcw.inv();
//    pointCloudGenerator->PushFrame(frame); //OnKeyFrameAvailable(frame);

    pointCloudGenerator->Reproject(frame.imRGB, frame.imDepth, frame.mTcw);

    system_clock::time_point endTime = system_clock::now();
    auto duration = duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cout << "每帧处理时间为　" << (float)duration * microseconds::period::num / microseconds::period::den << "s"<< endl;

}

void ProduceItem(ItemRepository *ir, const ark::RGBDFrame& item) {
    std::unique_lock<std::mutex> lock(ir->mtx);
    while(ir->item_buffer.size() == kItemsToProduce) { // item buffer is full, just wait here.
        std::cout << "the number of \n";
        (ir->repo_not_full).wait(lock); // 生产者等待"产品库缓冲区不为满"这一条件发生.
    }

    ir->item_buffer.push(item);
    (ir->repo_not_empty).notify_all(); // 通知消费者产品库不为空.
    lock.unlock(); // 解锁.代码结束后自动解锁
}

ark::RGBDFrame ConsumeItem(ItemRepository *ir) {

    std::unique_lock<std::mutex> lock(ir->mtx);
    // item buffer is empty, just wait here.
    while(ir->item_buffer.empty()) {
        std::cout << "当前数据列表为空\n";
        (ir->repo_not_empty).wait(lock); // 消费者等待"产品库缓冲区不为空"这一条件发生.
    }

    ark::RGBDFrame data((ir->item_buffer).front()); // 读取某一产品
    (ir->item_buffer).pop();
    (ir->repo_not_full).notify_all(); // 通知消费者产品库不为满.
    lock.unlock(); // 解锁.

    return std::move(data); // 返回产品.
}

ark::RGBDFrame ConsumeRTItem(ItemRepository *ir) {
    std::unique_lock<std::mutex> lock(ir->mtx);

    while(ir->item_buffer.empty()) {
        std::cout << "当前数据列表为空\n";
        (ir->repo_not_empty).wait(lock);
    }

}

/*
int ProducerTask(){

    int camNum = 1;

    Status rc;
    rc = openni::OpenNI::initialize();
    if(rc != STATUS_OK) {
        printf("init failed:%s\n", OpenNI::getExtendedError());
        return 1;
    }

    rc = device.open(ANY_DEVICE);
    if(rc != STATUS_OK) {
        printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
        return 2;
    }

    if (device.getSensorInfo(SENSOR_DEPTH) != NULL)
    {
        rc = depth.create(device, SENSOR_DEPTH);
        if (rc != STATUS_OK)
        {
            printf("Couldn't create depth stream\n%s\n", OpenNI::getExtendedError());
            return 3;
        }
    }

    VideoMode videoMode = depth.getVideoMode();
    videoMode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
    videoMode.setResolution(640, 480);
    videoMode.setFps(30);
    depth.setVideoMode(videoMode);

    rc = depth.start();
    if (rc != STATUS_OK)
    {
        printf("Couldn't start the depth stream\n%s\n", OpenNI::getExtendedError());
        return 4;
    }
    rc = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
    if(rc != STATUS_OK) {
        printf("Couldn't Set 对齐\n%s\n", OpenNI::getExtendedError());
        return 2;
    }

    const SensorInfo* colorSensorInfo = device.getSensorInfo(openni::SENSOR_COLOR);

    if (colorSensorInfo != NULL){
        isUvcCamera = false;
        rc = color.create(device, SENSOR_COLOR);
        if (rc != STATUS_OK) {
            printf("Couldn't create ir stream\n%s\n", OpenNI::getExtendedError());
            return 3;
        }

        rc = color.start();
        if (rc != STATUS_OK) {
            printf("Couldn't start the ir stream\n%s\n", OpenNI::getExtendedError());
            return 4;
        }
    }

    if (isUvcCamera && openAllStream) {

        if (device.getSensorInfo(SENSOR_IR) != NULL)
        {
            rc = ir.create(device, SENSOR_IR);
            if (rc != STATUS_OK)
            {
                printf("Couldn't create ir stream\n%s\n", OpenNI::getExtendedError());
                return 3;
            }
        }

        rc = ir.start();
        if (rc != STATUS_OK)
        {
            printf("Couldn't start the ir stream\n%s\n", OpenNI::getExtendedError());
            return 4;
        }

    }

    if (isUvcCamera) {
        capture.set(6, CV_FOURCC('M', 'J', 'P', 'G'));
        if (!capture.open(1))
        {
            capture.open(0);
        }

        if (!capture.isOpened())
        {
            return -1;
        }
    }


    const char* title = "UVC Color";
    cvNamedWindow(title, 1);
    int cnt = 0;
    char fpsStr[64] = "30.0";
//  最大上限问题在p
    int frameNum = 1000;
    int realNum = 0;
    while(cnt < frameNum){
        Mat previewImg(480, 640, CV_8UC3);
        waitForFrame(previewImg);
        IplImage image = previewImg;
        if(cnt > 10 && cnt % 5 == 0){
            cv::Mat floatMat;
            depthRaw.convertTo(floatMat, CV_32FC1);
            ark::RGBDFrame tem(colorImg, floatMat, realNum++);
            ProduceItem(&gItemRepository, tem);
        }

        cnt++;
        cvShowImage(title, &image);
        int key = cvWaitKey(10);
        if(key >= 0) {
            break;
        }
    }
    if (isUvcCamera) {
        capture.release();
    }
    device.close();
    openni::OpenNI::shutdown();

    return 0;
}
*/




// 生产者任务
void ProducerTask() {
    for (int i = 0; i < kItemsToProduce; ++i) {
        ark::RGBDFrame frame = saveFrame->frameLoad(i);
        ProduceItem(&gItemRepository, frame); // 循环生产 kItemsToProduce 个产品.
    }
}


void ConsumerTask() // 消费者任务
{
    static int cntFrame = 0;
    while(1) {
        ark::RGBDFrame item = ConsumeItem(&gItemRepository); // 消费一个产品.
        printf("current frame id is %d\n", item.frameId);
        FusionPart(item, cntFrame);

        if (++cntFrame == kItemsToProduce) break; // 如果产品消费个数为 kItemsToProduce, 则退出.
    }
}



void application_thread() {

    std::thread producer(ProducerTask); // 创建生产者线程.
    std::thread consumer(ConsumerTask); // 创建消费之线程.
    producer.join();
    consumer.join();
    //需要研究判断此处添加join是否合理。
//    int tframe = 0;
//    using namespace std::chrono;
//    system_clock::time_point startTime = system_clock::now();
    // clock_t t1 = clock();

//    while (tframe < 10) {
//        printf("The tframe is %d\n",tframe);
//        //当前按照文件的读取方法
//        ark::RGBDFrame frame = saveFrame->frameLoad(tframe);
//
//        ICP->CVTimage2Point(tframe, frame.imDepth);
//
//        Eigen::Matrix4f show = ICP->getRT();
//        cout << ICP->frameID << endl;
//        cout << ICP->currentFrame->size()<<endl;
//        cout << ICP->lastFrame->size() << endl;
//        cout << "last RT is \n" << ICP->lastMat << endl;
//        cout << "current RT is \n" << show << endl;
//        tframe += 1;
//        Eigen::Matrix4f showInv = show.inverse();
//        cv::eigen2cv(showInv, frame.mTcw);
//        cout << frame.mTcw << endl;
//
//
//        if(frame.frameId == -1){
//            empty ++;
//            continue;
//        }
//
//        cv::cvtColor(frame.imRGB, frame.imRGB, cv::COLOR_BGR2RGB);
//        cv::Mat Twc = frame.mTcw.inv();
//        // pointCloudGenerator->Reproject(frame.imRGB, frame.imDepth, Twc);
//        pointCloudGenerator->PushFrame(frame); //OnKeyFrameAvailable(frame);
//    }
//    system_clock::time_point endTime = system_clock::now();
//    auto duration = duration_cast<std::chrono::microseconds>(endTime - startTime).count();
//    cout << "总的时间消耗为　" << (float)duration * microseconds::period::num / microseconds::period::den << "s"<< endl;
}

void draw_box(float ox, float oy, float oz, float width, float height, float length) {
    glLineWidth(1.0f);
    glColor3f(1.0f, 1.0f, 1.0f);

    glBegin(GL_LINES);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox + width, oy, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy + height, oz);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox + width, oy + height, oz + length);

    glVertex3f(ox + width, oy + height, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox + width, oy + height, oz + length);

    glEnd();
}

void draw_origin(float length) {
    glLineWidth(1.0f);
    glColor3f(1.0f, 1.0f, 1.0f);

    glBegin(GL_LINES);

    glVertex3f(0.f,0.f,0.f);
    glVertex3f(length,0.f,0.f);

    glVertex3f(0.f,0.f,0.f);
    glVertex3f(0.f,length,0.f);

    glVertex3f(0.f,0.f,0.f);
    glVertex3f(0.f,0.f,length);

    glEnd();
}


void init() {
    glewInit();

    glViewport(0, 0, window_width, window_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluLookAt(0,20,0,0,0,0,0,1,0);
    gluPerspective(45.0, (float) window_width / window_height, 10.0f, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -3.0f);
    glRotatef(90,1.0,0.0,0.0);

}

void display_func() {
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glPushMatrix();

    if (buttonState == 1) {
        xRot += (xRotLength - xRot) * 0.1f;
        yRot += (yRotLength - yRot) * 0.1f;
    }

    glTranslatef(xTrans, yTrans, zTrans);
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);
    // printf("display_fun running!!!!");
    pointCloudGenerator->Render();

    draw_origin(4.f);

    glPopMatrix();
    glutSwapBuffers();

}

void idle_func() {
    //空闲时间不做任何处理。
//    glutPostRedisplay();
}

void reshape_func(GLint width, GLint height) {
    window_width = width;
    window_height = height;

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(45.0, (float) width / height, 0.001, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // glRotatef(90,0.0,1.0,0.0);

    glTranslatef(0.0f, 0.0f, -3.0f);
}

int countFiles(string filename){
    DIR *dp;
    int i = 0;
    struct dirent *ep;
    dp = opendir (filename.c_str());

    if (dp != NULL) {
        while (ep = readdir (dp)) {
            i++;
        }
        (void) closedir (dp);
    }
    else {
        perror ("Couldn't open the directory");
    }

    i -= 2;
    printf("There's %d files in the current directory.\n", i);
    return i;
}


void keyboard_func(unsigned char key, int x, int y) {
    if (key == ' ') {
        if (!stop) {
            app = new thread(application_thread);
            stop = !stop;
        } else {
//            slam->RequestStop();
            pointCloudGenerator->RequestStop();
//            bridgeRSD435->Stop();
        }
    }

    if (key == 'w') {
        zTrans += 0.3f;
    }

    if (key == 's') {
        zTrans -= 0.3f;
    }

    if (key == 'a') {
        xTrans += 0.3f;
    }

    if (key == 'd') {
        xTrans -= 0.3f;
    }

    if (key == 'q') {
        yTrans -= 0.3f;
    }

    if (key == 'e') {
        yTrans += 0.3f;
    }

    if (key == 'p') {
//        slam->RequestStop();
        pointCloudGenerator->RequestStop();
//        bridgeRSD435->Stop();

        pointCloudGenerator->SavePly("model.ply");
    }

    if (key == 'v')
        wireframe = !wireframe;


    glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        buttonState = 1;
    } else if (state == GLUT_UP) {
        buttonState = 0;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}

void motion_func(int x, int y) {
    float dx, dy;
    dx = (float) (x - ox);
    dy = (float) (y - oy);

    if (buttonState == 1) {
        xRotLength += dy / 5.0f;
        yRotLength += dx / 5.0f;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}

int main(int argc, char **argv) {
    std::cout << "here" << std::endl;
    if (argc != 3) {
        cerr << endl << "Usage: ./load_frames path_to_frames path_to_settings" << endl;
        return 1;
    }
    std::cout << "here" << std::endl;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    (void) glutCreateWindow("GLUT Program");
    std::cout << "here" << std::endl;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    pointCloudGenerator = new ark::PointCloudGenerator(argv[2]);
//    slam = new ark::ORBSLAMSystem(argv[1], argv[2], ark::ORBSLAMSystem::RGBD, true);
//    bridgeRSD435 = new BridgeRSD435();
    std::cout << "here" << std::endl;
    saveFrame = new ark::SaveFrame(argv[1]);
    std::cout << "here" << std::endl;
    ICP = new ark::ICPPart();
    ORBAlignment = new ark::orbAlignment();
    //初始化ICP部分
//    ICP = new ark::ICPPart();

//    slam->AddKeyFrameAvailableHandler([pointCloudGenerator](const ark::RGBDFrame &keyFrame) {
//        return pointCloudGenerator->OnKeyFrameAvailable(keyFrame);
//    }, "PointCloudFusion");

    init();

    glutSetWindowTitle("OpenARK 3D Reconstruction");
    glutDisplayFunc(display_func);
    glutReshapeFunc(reshape_func);
    glutIdleFunc(idle_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutKeyboardFunc(keyboard_func);
    glutMainLoop();

    delete pointCloudGenerator;
//    delete slam;
//    delete bridgeRSD435;
    delete saveFrame;
    delete app;
    delete ICP;
    delete ORBAlignment;

    return EXIT_SUCCESS;
}
