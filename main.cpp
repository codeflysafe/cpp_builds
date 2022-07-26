#include <iostream>
#include "yolox/yolox.h"
int main()
{
    cout << "hello world !" << endl;
    yolox net("/Users/sjhuang/Documents/projects/cpp_builds/models/yolox_s.onnx", 0.25, 0.45);
    string imgpath = "/Users/sjhuang/Documents/projects/cpp_builds/asset/WechatIMG2281.jpeg";  ///输入图片的路径，你也可以改成外部传参argv的方式，或者是读取视频文件
    Mat src_img = imread(imgpath);
    cout << src_img.size() << endl;
    cout << "start detecting image !" << endl;
    net.detect(src_img);
    cout << "detect image success !" << endl;
    imwrite("/Users/sjhuang/Documents/projects/cpp_builds/asset/res.png",src_img);
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
    imshow(kWinName, src_img);
    waitKey(1);
    destroyAllWindows();
}