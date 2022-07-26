//
// Created by sjhuang on 2022/7/25.
//

#include "yolox.h"

/**
 * yolox 构建类别
 * @param model_path: 模型路径
 * @param conf_thr: 置信度阈值
 * @param nms_thr：nms 阈值
 */
yolox::yolox(string model_path, float conf_thr, float nms_thr) {
    this->prob_thr = conf_thr;
    this->nms_thr = nms_thr;
//    this->model_path = model_path;

    ifstream ifs(this->classesFile.c_str());
    string line;
    while(getline(ifs, line)){
        this->classes.push_back(line);
    }
    this->num_class = int(this->classes.size());
    this->net = readNet(model_path);
}


/**
 * 等比例缩放
 * @param src_img : source image
 * @param scale : 比例
 * @return
 */
Mat yolox::resize_img(Mat &src_img, float *scale) {

    float r = min(this->input_shapes[1] /(src_img.cols * 1.0), this->input_shapes[0] /(src_img.rows * 1.0));
    *scale = r;

    int new_h = r * src_img.cols;
    int new_w = r * src_img.rows;

    // 缩放图片
    Mat re(new_h, new_w, CV_8UC3);
    resize(src_img, re,re.size());

    // copy到原输入大小
    Mat out(this->input_shapes[1], this->input_shapes[0], CV_8UC3, Scalar(114,114,114));
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
    return out;
}

/**
 * 标准化，归一化操作
 * @param src_img : source image
 */
void yolox::normalize(Mat &src_img) {
    //
    cvtColor(src_img,src_img, cv::COLOR_BGR2RGB);
    src_img.convertTo(src_img, CV_32F);
    int i = 0, j = 0;
    for(; i < src_img.rows; i++){
        float* pdata = (float*)(src_img.data + i * src_img.step);
        for(; j < src_img.cols; j++){
            // 三通道
            pdata[0] = (pdata[0] / 255.0 - this->mean[0]) / this->std[0];
            pdata[1] = (pdata[1] / 255.0 - this->mean[1]) / this->std[1];
            pdata[2] = (pdata[2] / 255.0 - this->mean[2]) / this->std[2];
            // 指针前进
            pdata += 3;
        }
    }
}

int yolox::get_max_class(float* scores)
{
    float max_class_socre = 0, class_socre = 0;
    int max_class_id = 0, c = 0;
    for (c = 0; c < this->num_class; c++) //// get max socre
    {
        if (scores[c] > max_class_socre)
        {
            max_class_socre = scores[c];
            max_class_id = c;
        }
    }
    return max_class_id;
}

/**
 * 检测图片
 * @param src_img 原图片
 */
void yolox::detect(Mat &src_img) {
    float scale = 1.0;
    // resize img
    Mat img = this->resize_img(src_img, &scale);
    // normalize the source image
    this->normalize(img);
    Mat blob = blobFromImage(img);

    this->net.setInput(blob);
    vector<Mat> outs;
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
    // channel == 3
    if(outs[0].dims == 3){
        // 候选框数量
        const int num_proposals = outs[0].size[1];
        outs[0] = outs[0].reshape(0, num_proposals);
    }

    // generate proposals, decode outputs
    vector<int> class_ids;
    vector<float> confids;
    vector<Rect> boxes;

//    float ratio_h = (float)src_img.rows / this->input_shapes[0], ratio_w = (float)src_img.cols / this->input_shapes[1];
    int n = 0, i = 0, j = 0, n_out = this->classes.size() + 5, row_ind = 0;
    float *pdata = (float *)outs[0].data;
    for (n = 0; n < 3; n++)   ///尺度
    {
        const int num_grid_x = (int)(this->input_shapes[1] / this->strides[n]);
        const int num_grid_y = (int)(this->input_shapes[0] / this->strides[n]);
        for (i = 0; i < num_grid_y; i++)
        {
            for (j = 0; j < num_grid_x; j++)
            {
                float box_score = pdata[4];

                //int class_idx = this->get_max_class(pdata + 5);
                Mat scores = outs[0].row(row_ind).colRange(5, outs[0].cols);
                Point classIdPoint;
                double max_class_socre;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                int class_idx = classIdPoint.x;

                float cls_score = pdata[5 + class_idx];
                float box_prob = box_score * cls_score;
                if (box_prob > this->prob_thr)
                {
                    float x_center = (pdata[0] + j) * this->strides[n];
                    float y_center = (pdata[1] + i) * this->strides[n];
                    float w = exp(pdata[2]) * this->strides[n];
                    float h = exp(pdata[3]) * this->strides[n];
                    float x0 = x_center - w * 0.5f;
                    float y0 = y_center - h * 0.5f;

                    class_ids.push_back(class_idx);
                    confids.push_back(box_prob);
                    boxes.push_back(Rect(int(x0), int(y0), (int)(w), (int)(h)));
                }

                pdata += n_out;
                row_ind++;
            }
        }
    }
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confids, this->prob_thr, this->nms_thr, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        // adjust offset to original unpadded
        float x0 = (box.x) / scale;
        float y0 = (box.y) / scale;
        float x1 = (box.x + box.width) / scale;
        float y1 = (box.y + box.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(src_img.cols - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(src_img.rows - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(src_img.cols - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(src_img.rows - 1)), 0.f);

        rectangle(src_img, Point(x0, y0), Point(x1, y1), Scalar(0, 0, 255), 2);
        //Get the label for the class name and its confidence
        string label = format("%.2f", confids[idx]);
        label = this->classes[class_ids[idx]] + ":" + label;
        //Display the label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        y0 = std::max(y0, (float)labelSize.height);
        //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
        putText(src_img, label, Point(x0, y0), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
    }
}

