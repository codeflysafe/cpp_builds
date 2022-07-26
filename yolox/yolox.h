//
// Created by sjhuang on 2022/7/25.
//

#ifndef CPP_BUILDS_YOLOX_H
#define CPP_BUILDS_YOLOX_H
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace dnn;
using namespace std;

//class yolox {
//public:
//    yolox(string model_path, float conf_thr, float nms_thr);
//    void detect(Mat &src_img);
//
//private:
//    const int strides[3] = {8, 16, 32};
//    ////这个是存放COCO数据集的类名，如果你是用自己数据集训练的，那么需要修改
//    const string classesFile = "/Users/sjhuang/Documents/projects/cpp_builds/yolox/coco.classes";
//    const int input_shapes[2] = {640, 640}; // h, w
//    // image net 均值和标准差
//    const float mean[3] = { 0.485, 0.456, 0.406 };
//    const float std[3] = { 0.229, 0.224, 0.225 };
////    string model_path;
//    float prob_thr;
//    float nms_thr;
//    vector<string> classes;
//    int num_class;
//    Net net;
//
//    Mat resize_img(Mat &src_img, float *scale);
//    void normalize(Mat &src_img);
//    int get_max_class(float  *scores);
//};

class yolox
{
public:
    yolox(string modelpath, float confThreshold, float nmsThreshold);
    void detect(Mat& srcimg);

private:
    const int stride[3] = { 8, 16, 32 };
    const string classesFile = "/Users/sjhuang/Documents/projects/cpp_builds/yolox/coco.classes";   ////这个是存放COCO数据集的类名，如果你是用自己数据集训练的，那么需要修改
    const int input_shape[2] = { 640, 640 };   //// height, width
    const float mean[3] = { 0.485, 0.456, 0.406 };
    const float std[3] = { 0.229, 0.224, 0.225 };
    float prob_threshold;
    float nms_threshold;
    vector<string> classes;
    int num_class;
    Net net;

    Mat resize_image(Mat srcimg, float* scale);
    void normalize(Mat& srcimg);
    int get_max_class(float* scores);
};

#endif //CPP_BUILDS_YOLOX_H
