//
// Created by meruro on 2023/5/19.
//
#include<vector>
#include <fstream>

#include <iostream>
#include <numeric>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2\imgproc.hpp>
#include<opencv2\imgproc\types_c.h>
#include <NIVisionExtLib.h>
#include "NIVisionExtExports.h"

#define PI 3.1415926
//弧度转角度
#define r2a(x) ((x)*180/PI)
//角度转弧度
#define a2r(x) ((x)*PI/180)

//日志
//#include "mylog.h"
//#include "spdlog/sinks/basic_file_sink.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

// 类别
vector<string> class_list;
// 权重
Net net;

ofstream outfile;

// Draw the predicted bounding box.
void draw_label(Mat &input_image, string label, int left, int top) {
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

double getSeconds(chrono::time_point<chrono::system_clock> &start,
                  chrono::time_point<chrono::system_clock> &end) {
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    return double(duration.count()) / 1000000;
}

vector<Mat> pre_process(Mat &input_image) {
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

Mat post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name) {
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *) outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) {
                // Store class ID and confidence in the pre-defined respective vectors.

                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
}

EXTERN_C void NI_EXPORT load_class_list(char *path)
//void load_class_list(const string &path)
{
//    outfile.open("D:/afile0.txt");

//    outfile << *path << endl;
    class_list.clear();
    std::ifstream ifs(path);
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
//        outfile << line << endl;
    }
}

EXTERN_C void NI_EXPORT load_net(char* path, int is_cuda)
//void load_net(const string &path,int is_cuda)
{
//    outfile << is_cuda << endl;
    net = cv::dnn::readNet(path);
    if (is_cuda) {
//        std::cout << "Attempty to use CUDA\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
//        outfile << "Running on CUDA\n" << endl;
    } else {
//        std::cout << "Running on CPU\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
//        outfile << "Running on CPU\n" << endl;
    }
}


EXTERN_C void NI_EXPORT
detect_all(NIImageHandle sourceHandle_src, NIImageHandle destHandle, NIErrorHandle errorHandle, double *time) {
    //auto file_logger = spdlog::basic_logger_mt("basic_logger", "D:/basic.txt");
    //spdlog::set_default_logger(file_logger);
    NIERROR error = NI_ERR_SUCCESS;
    ReturnOnPreviousError(errorHandle);
    try {
//        outfile.open("D:/afile2.txt");
        if (!sourceHandle_src || !destHandle || !errorHandle) {
            ThrowNIError(NI_ERR_NULL_POINTER);
        }
        NIImage source_src(sourceHandle_src);
        NIImage dest(destHandle);

        cv::Mat sourceMat_src;
        cv::Mat destMat;
        // ni图片转Mat
        ThrowNIError(source_src.ImageToMat(sourceMat_src));
//        outfile << sourceMat_src.shape << endl;
        if (source_src.type == NIImage_RGB32) {
            cv::cvtColor(sourceMat_src, sourceMat_src, CV_RGB2BGR);
            //outfile << "success" << endl;
//            imwrite("D:/haha1.png", sourceMat_src);

        }
//        cv::imwrite("D:/srcimg.png",sourceMat_src);
        outfile << source_src.type << endl;
        auto start = chrono::system_clock::now(); // 开始时间
//        outfile << "时间" << endl;

        vector<Mat> detections;
//        detections = pre_process(sourceMat_src);
        Mat blob;
        blobFromImage(sourceMat_src, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), false, false);
//        outfile << "blob2" << endl;

        net.setInput(blob);
//        outfile << "blob2" << endl;
//        outfile << blob.size << endl;

        // Forward propagate.
        net.forward(detections, net.getUnconnectedOutLayersNames());
//        outfile << "pre_process" << endl;
        //推理后的图片处理
        Mat cloned_frame = sourceMat_src.clone();
        Mat img = post_process(cloned_frame, detections, class_list);

//        outfile << "post_process" << endl;
        // Put efficiency information.
        // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time : %.2f ms", t);
        putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
//        outfile << label << endl;
        auto end = chrono::system_clock::now(); // 结束时间
        *time = getSeconds(start, end);
        cv::cvtColor(img, destMat, CV_BGR2RGBA);

//        outfile << "success" << endl;

        ThrowNIError(dest.MatToImage(destMat));

    }
    catch (NIERROR &_err) {
        error = _err;
    }
    catch (std::string e) {
        outfile << e << endl;
        error = NI_ERR_OCV_USER;
    }
    ProcessNIError(error, errorHandle);
}


//int main()
//{
//    // Load class list.
//    load_class_list("../coco.names");
//
//
//    // Load image.
//    Mat frame;
//    frame = imread("../sample.jpg");
//
//    // Load model.
//    load_net("../models/yolov5s.onnx",false);
//    //图片推理
//    vector<Mat> detections;
//    detections = pre_process(frame);
//    //推理后的图片处理
//    Mat cloned_frame = frame.clone();
//    Mat img = post_process(cloned_frame, detections, class_list );
//
//    // Put efficiency information.
//    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
//
//    vector<double> layersTimes;
//    double freq = getTickFrequency() / 1000;
//    double t = net.getPerfProfile(layersTimes) / freq;
//    string label = format("Inference time : %.2f ms", t);
//    putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
//    cout << label << endl;
//
//    imshow("Output", img);
//    waitKey(0);
//
//    return 0;
//}