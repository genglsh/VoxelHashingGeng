//
// Created by gengshuai on 19-10-24.
//

#ifndef TSDF_VIDEOSET_H
#define TSDF_VIDEOSET_H
class VideoFrameDef {
public:

    void SetDepth(cv::Mat Img) {
        depthImg = Img;
    }
    void SetRGB(cv::Mat Img) {
        rgbImg = Img;
    }

    cv::Mat depthImg;
    cv::Mat rgbImg;
};

class VideoStreamDef {
public:
    VideoStreamDef() {
        queue<VideoFrameDef>* tem = new queue<VideoFrameDef>();
        OBVideoqueue = shared_ptr<queue<VideoFrameDef>>(tem);
    }

    bool AppendFrame(const VideoFrameDef& frame){
        if(OBVideoqueue->size() < 100) {
            OBVideoqueue->push(frame);
            return true;
        }
        return false;
    }

    VideoFrameDef GetFrame(){
        if(!OBVideoqueue->empty()){
            OBVideoqueue->pop();
        }
    }

private:
    std::shared_ptr <queue<VideoFrameDef>> OBVideoqueue;
};

#endif //TSDF_VIDEOSET_H
