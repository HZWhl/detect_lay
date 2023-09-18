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

#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file
#include<opencv2\imgproc.hpp>
#include<opencv2\imgproc\types_c.h>
#include <NIVisionExtLib.h>
#include "NIVisionExtExports.h"

std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255) , cv::Scalar(0, 255, 0) , cv::Scalar(255, 0, 0) ,
                                   cv::Scalar(255, 100, 50) , cv::Scalar(50, 100, 255) , cv::Scalar(255, 50, 100) };
std::vector<std::string> class_names;

using namespace cv;
using namespace dnn;

// Keep the ratio before resize
Mat letterbox(const cv::Mat& source)
{
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
std::shared_ptr<ov::CompiledModel> compiled_model;
// 用ov::Core::compile_model()方法创建对象
// 释放compiled_model

EXTERN_C void NI_EXPORT load_net(char* path, int is_cuda)
{
// -------- Step 2. Compile the Model --------
    std::string device;
    if(is_cuda){
        device = "GPU";
    }else{
        device = "CPU";
    };
    ov::Core core;
    compiled_model = std::make_shared<ov::CompiledModel>(core.compile_model(path, device));
//    compiled_model = core.compile_model(path, device);
// -------- Step 3. Create an Inference Request --------

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
detect_all(NIImageHandle sourceHandle_src, NIImageHandle destHandle, NIErrorHandle errorHandle,double *score_threshold, double *time,int32_t *exist) {
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
        Mat letterbox_img = letterbox(sourceMat_src);
        float scale = letterbox_img.size[0] / 640.0;
        Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), false);
//        cv::imwrite("D:/aa.png",sourceMat_src);/
        // -------- Step 5. Feed the blob into the input node of the Model -------
        // Get input port for model with one input
        // Create tensor from external memory
        ov::InferRequest infer_request = compiled_model->create_infer_request();
        auto input_port = compiled_model->input();
        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));

        // Set input tensor for model with one input
        infer_request.set_input_tensor(input_tensor);

        // -------- Step 6. Start inference --------
        infer_request.infer();

        // -------- Step 7. Get the inference result --------
        auto output = infer_request.get_output_tensor(0);
        auto output_shape = output.get_shape();
    //    std::cout << "The shape of output tensor:" << output_shape << std::endl;
    //    int rows = output_shape[2];        //8400
    //    int dimensions = output_shape[1];  //84: box[cx, cy, w, h]+80 classes scores

        // -------- Step 8. Postprocess the result --------
        float* data = output.data<float>();
        Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
        transpose(output_buffer, output_buffer); //[8400,84]
//        float score_threshold = 0.25;
        float nms_threshold = 0.5;
        std::vector<int> class_ids;
        std::vector<float> class_scores;
        std::vector<Rect> boxes;

        // Figure out the bbox, class_id and class_score
        for (int i = 0; i < output_buffer.rows; i++) {
            Mat classes_scores = output_buffer.row(i).colRange(4, 5);
            Point class_id;
            double maxClassScore;
            minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > score_threshold) {
                class_scores.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                float cx = output_buffer.at<float>(i, 0);
                float cy = output_buffer.at<float>(i, 1);
                float w = output_buffer.at<float>(i, 2);
                float h = output_buffer.at<float>(i, 3);

                int left = int((cx - 0.5 * w) * scale);
                int top = int((cy - 0.5 * h) * scale);
                int width = int(w * scale);
                int height = int(h * scale);

                boxes.push_back(Rect(left, top, width, height));
            }
        };
        //NMS
        std::vector<int> indices;
        NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);

        // -------- Visualize the detection results -----------
        for (size_t i = 0; i < indices.size(); i++) {
            int index = indices[i];
            int class_id = class_ids[index];
            rectangle(sourceMat_src, boxes[index], colors[class_id % 6], 2, 8);
            std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]).substr(0, 4);
            Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
            Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height+5);
            cv::rectangle(sourceMat_src, textBox, colors[class_id % 6], FILLED);
            exist[class_id] = exist[class_id]+1;
            putText(sourceMat_src, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
        }
//        outfile << *result_list.data() << endl;
//        int32_t arr[class_names.size()];
//        exist = result_list.data();
        auto end = chrono::system_clock::now(); // 结束时间
        *time = getSeconds(start, end);
        cv::cvtColor(sourceMat_src, destMat, CV_BGR2RGBA);

//        outfile << "success" << endl;

        ThrowNIError(dest.MatToImage(destMat));

    }
    catch (NIERROR &_err) {
        error = _err;
    }
    catch (std::string e) {
//        outfile << e << endl;
        error = NI_ERR_OCV_USER;
    }
    ProcessNIError(error, errorHandle);
}

EXTERN_C void NI_EXPORT release_model()
{
    compiled_model = nullptr;
}