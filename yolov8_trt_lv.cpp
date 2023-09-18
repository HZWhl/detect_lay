//
// Created by meruro on 2023/5/24.
//

//
// Created by meruro on 2023/5/20.
//
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

//#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file
#include<opencv2\imgproc.hpp>
#include<opencv2\imgproc\types_c.h>
#include <NIVisionExtLib.h>
#include "NIVisionExtExports.h"
#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"
//#include "infer.cu"
//#include "yolo.cu"


std::vector<cv::Scalar> colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                                  cv::Scalar(255, 100, 50), cv::Scalar(50, 100, 255), cv::Scalar(255, 50, 100)};
std::vector<std::string> class_names;

using namespace cv;
using namespace dnn;

// Keep the ratio before resize
Mat letterbox(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

double getSeconds(chrono::time_point<chrono::system_clock> &start,
                  chrono::time_point<chrono::system_clock> &end) {
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    return double(duration.count()) / 1000000;
}

//ov::CompiledModel compiled_model;

// 定义一个智能指针指向ov::CompiledModel对象
std::shared_ptr<yolo::Infer> net;
// 用ov::Core::compile_model()方法创建对象
// 释放compiled_model

// 图片转换函数
yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

EXTERN_C void NI_EXPORT load_net(char *path, double *score_threshold) {

    float confidence_threshold = 0.25f;
    float nms_threshold = 0.5f;
    net = yolo::load(path, yolo::Type::V8, confidence_threshold, nms_threshold);
}
EXTERN_C void NI_EXPORT load_class_list(char *path)
//void load_class_list(const string &path)
{

    class_names.clear();
    std::ifstream ifs(path);
    std::string line;
    while (getline(ifs, line)) {
        class_names.push_back(line);
    }
}

EXTERN_C void NI_EXPORT
detect_all(NIImageHandle sourceHandle_src, NIImageHandle destHandle, NIErrorHandle errorHandle,
           double *time, int32_t *exist) {
//    auto file_logger = spdlog::basic_logger_mt("basic_logger", "D:/basic.txt");
//spdlog::set_default_logger(file_logger);
    NIERROR error = NI_ERR_SUCCESS;
    ReturnOnPreviousError(errorHandle);
    try {
//        ofstream outfile;
//        outfile.open("D:/afile1000.txt");
        if (!sourceHandle_src || !destHandle || !errorHandle) {
            ThrowNIError(NI_ERR_NULL_POINTER);
        }
        NIImage source_src(sourceHandle_src);
        NIImage dest(destHandle);

        cv::Mat sourceMat_src;
        cv::Mat destMat;
// ni图片转Mat
        ThrowNIError(source_src.ImageToMat(sourceMat_src));
        cv::cvtColor(sourceMat_src, sourceMat_src, CV_RGB2BGR);
//        cv::imwrite("D:/srcimg.png",sourceMat_src);
//        outfile << source_src.type << endl;
        auto start = chrono::system_clock::now(); // 开始时间
// Preprocess the image


//        if (net == nullptr) outfile << "no load net" << endl;

        auto objs = net->forward(cvimg(sourceMat_src));
//        outfile << "forward!" << endl;

        for (auto &obj : objs) {
//            outfile << "1111--" << endl;
            uint8_t b, g, r;
            tie(b, g, r) = yolo::random_color(obj.class_label);
            cv::rectangle(sourceMat_src, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                          cv::Scalar(b, g, r), 5);

            auto name = class_names[int(obj.class_label)];
            auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
            int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            exist[obj.class_label] = exist[obj.class_label]+1;
            cv::rectangle(sourceMat_src, cv::Point(obj.left - 3, obj.top - 33),
                          cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(sourceMat_src, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        auto end = chrono::system_clock::now(); // 结束时间
        *time = getSeconds(start, end);
        cv::cvtColor(sourceMat_src, destMat, CV_BGR2RGBA);

        ThrowNIError(dest.MatToImage(destMat));

    }
    catch (NIERROR &_err) {
        error = _err;
    }
    catch (std::string e) {
        error = NI_ERR_OCV_USER;
    }
    ProcessNIError(error, errorHandle);
}

EXTERN_C void NI_EXPORT release_model() {
    net = nullptr;
}