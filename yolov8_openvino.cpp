//
// Created by meruro on 2023/5/20.
//
#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file

std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255) , cv::Scalar(0, 255, 0) , cv::Scalar(255, 0, 0) ,
                                   cv::Scalar(255, 100, 50) , cv::Scalar(50, 100, 255) , cv::Scalar(255, 50, 100) };
//const std::vector<std::string> class_names = {
//        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//        "hair drier", "toothbrush" };

const std::vector<std::string> class_names = {"ds",};
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

int main(int argc, char* argv[])
{
    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    std::cout << "start" << std::endl;
//    const std::string base_dir = "E:\\0.GSS\\Labview\\fan_wei\\bin\\";
    const std::string base_dir = "D:\\projects\\cppProjects\\detect_lay\\";
    // -------- Step 2. Compile the Model --------
    auto compiled_model = core.compile_model(base_dir+"best_all_openvino_model\\best_all.xml", "CPU");

    std::cout << "start2" << std::endl;
    // -------- Step 3. Create an Inference Request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    std::cout << "start3" << std::endl;
    // -------- Step 4.Read a picture file and do the preprocess --------
    Mat img = cv::imread(base_dir+"Pic_2023_05_16_103736_blockId#14044.bmp");
    // Preprocess the image
    Mat letterbox_img = letterbox(img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);

    std::cout << "start5" << std::endl;
    // -------- Step 6. Start inference --------
    infer_request.infer();

    // -------- Step 7. Get the inference result --------
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();
    std::cout << "The shape of output tensor:" << output_shape << std::endl;
//    int rows = output_shape[2];        //8400
//    int dimensions = output_shape[1];  //84: box[cx, cy, w, h]+80 classes scores

    std::cout << "start6" << std::endl;
    // -------- Step 8. Postprocess the result --------
    float* data = output.data<float>();
    std::cout << output_shape[1]<<'\n' << output_shape[2]<<'\n'<<*data << std::endl;
    Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,84]
    float score_threshold = 0.25;
    float nms_threshold = 0.5;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    std::vector<int> result_list(class_names.size(),0);
    // Figure out the bbox, class_id and class_score
    for (int i = 0; i < output_buffer.rows; i++) {
        Mat classes_scores = output_buffer.row(i).colRange(4, 4+class_names.size());
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
        rectangle(img, boxes[index], colors[class_id % 6], 2, 8);
        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]).substr(0, 4);
        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height+5);
        cv::rectangle(img, textBox, colors[class_id % 6], FILLED);

        result_list[class_id] = result_list[class_id]++;
        putText(img, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
    }
//    std::cout <<"结果数组数量: " <<*class_ids.data()<< "   "<<class_ids.size() <<std::endl;
    std::cout <<"结果数组数量: " <<*result_list.data()<< "   "<<result_list.size() <<std::endl;
    std::cout << "v = ";
    for (int i = 0; i < result_list.size(); i++) // 使用for循环遍历vector中的每个元素
    {
        std::cout << result_list[i] << " "; // 使用下标运算符[]访问并输出vector中的元素
    }
    std::cout << std::endl;

    namedWindow("YOLOv8 OpenVINO Inference C++ Demo", WINDOW_AUTOSIZE);
    imshow("YOLOv8 OpenVINO Inference C++ Demo", img);
    waitKey(0);
    destroyAllWindows();
    return 0;
}