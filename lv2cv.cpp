//
// Created by zfw on 2021/9/24.
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

//typdefs
typedef cv::Point2f PointFloat;
typedef cv::Point2d PointDouble;

#define OCV_RGB CV_8UC3

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;

// 轮廓
//vector<vector<Point>> contours;
//// 可选的输出向量(std::vector)，包含了图像的拓扑信息，
//vector<Vec4i> hierarchy;
//vector<vector<Point>> best_contours;
//void on_Matching(int, void*);

double getSeconds(chrono::time_point<chrono::system_clock>& start,
    chrono::time_point<chrono::system_clock>& end) {
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    return double(duration.count()) / 1000000;
}

//去除离群值
std::vector<cv::Point> Mad(std::vector<Point> nums, double& mean) {
    int n = nums.size();
    int sum = 0;
    double accum = 0;
    //    if (n<2)
    //    {
    //        return nums[0];
    //    }
    for (int i = 0; i < nums.size(); i++) {
        sum = sum + nums[i].y;
        //        cout<<i<<" : "<<nums[i].y<<endl;
        //        accum += (d - mean)*(d - mean);
    }
    mean = sum / n;
    for (int i = 0; i < nums.size(); i++) {
        accum += (nums[i].y - mean) * (nums[i].y - mean);
        //        cout<<i<<" : "<<nums[i].x<<" :  "<<nums[i].x - mean <<endl;
    }
    double variance = accum / n; //方差
    double stdev = sqrt(variance); //标准差
    cout << "mean:" << mean << "  std:" << stdev << endl;
    std::vector<cv::Point> new_nums;
    double maxval = mean + stdev;
    double minval = mean - stdev;

    for (int i = 0; i < n; i++)
    {
        if (minval < nums[i].y && nums[i].y < maxval) {
            //            cout<<i<<" : "<<nums[i].x<<"y:"<<nums[i].y<<endl;
            new_nums.push_back(nums[i]);
        }
    }
    return new_nums;
}

bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
    //Number of key points
    int N = key_point.size();
    //构造矩阵X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            for (int k = 0; k < N; k++)
            {
                X.at<double>(i, j) = X.at<double>(i, j) +
                    std::pow(key_point[k].x, i + j);
            }
        }
    }
    //构造矩阵Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int k = 0; k < N; k++)
        {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) +
                std::pow(key_point[k].x, i) * key_point[k].y;
        }
    }
    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //求解矩阵A
    cv::solve(X, Y, A, cv::DECOMP_LU);
    return true;
}

vector<int> findPeaks(vector<double> num, int count)
{
    vector<int> sign;
    //  mypoints sign2;
    for (int i = 1; i < count; i++)
    {
        /*相邻值做差：
         *小于0，赋-1
         *大于0，赋1
         *等于0，赋0
         */
        double diff = num[i] - num[i - 1];
        if (diff > 0)
        {
            sign.push_back(1);
        }
        else if (diff < 0)
        {
            sign.push_back(-1);
        }
        else
        {
            sign.push_back(0);
        }
    }
    //再对sign相邻位做差
    //保存极大值和极小值的位置
    vector<int> indMax;
    vector<int> indMin;

    for (int j = 1; j < sign.size(); j++)
    {
        int diff = sign[j] - sign[j - 1];
        if (diff < 0)
        {
            indMax.push_back(j);
        }
        else if (diff > 0)
        {
            indMin.push_back(j);
        }
    }
    //cout << "max:" << endl;
    //for (int m = 0; m < indMax.size(); m++)
    //{
    //    cout << num[indMax[m]] << "  ";
    //}
    //cout << endl;
    //cout << "min:" << endl;
    //for (int n = 0; n < indMin.size(); n++)
    //{
    //    cout << num[indMin[n]] << "  ";
    //}
    return indMax;
}


// comparison Contour object
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
    double i = fabs(contourArea(cv::Mat(contour1)));
    double j = fabs(contourArea(cv::Mat(contour2)));
    return (i < j);
}

EXTERN_C void NI_EXPORT detect_lay(NIImageHandle sourceHandle_src, NIImageHandle destHandle, NIErrorHandle errorHandle, double* lay,double* time) {
    //auto file_logger = spdlog::basic_logger_mt("basic_logger", "D:/basic.txt");
    //spdlog::set_default_logger(file_logger);
    NIERROR error = NI_ERR_SUCCESS;
    ReturnOnPreviousError(errorHandle);
    ofstream outfile;
    vector<vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;
    vector<vector<cv::Point>> best_contours;
    vector<int> center_y(2);
    float rotated;
    float arr;
    try
    {
        if (!sourceHandle_src || !destHandle || !errorHandle) {
            ThrowNIError(NI_ERR_NULL_POINTER);
        }
        NIImage source_src(sourceHandle_src);
        NIImage dest(destHandle);

        Mat sourceMat_src;
        Mat destMat;
        // ni图片转Mat
        ThrowNIError(source_src.ImageToMat(sourceMat_src));
        // 如果是彩色图片,需要转换
        //spdlog::info("Welcome to spdlog!");
        //ilog->info("ss:{}",source_src.type);

        // 以写模式打开文件

        //outfile.open("D:/afile.txt");


        // 向文件写入用户输入的数据
        //outfile << source_src.type << endl;
        auto start = chrono::system_clock::now(); // 开始时间
        // 文字位置
        cv::Point org(10, 90);
        if (source_src.type == NIImage_RGB32)
        {
            cv::cvtColor(sourceMat_src, sourceMat_src, CV_BGRA2GRAY);
            //outfile << "success" << endl;
            //imwrite("D:/haha1.png", destMat);

        }
        transpose(sourceMat_src, sourceMat_src);
        flip(sourceMat_src, sourceMat_src, -1);

        adaptiveThreshold(sourceMat_src, destMat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, -2);

        Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
        erode(destMat, destMat, element);
        //imwrite("D:/haha2.png", destMat);
        findContours(destMat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        // sort contours
        //std::sort(contours.begin(), contours.end(), compareContourAreas);
        Mat drawImg = Mat::zeros(destMat.size(), CV_8UC3);
        drawImg.setTo(cv::Scalar(0, 0, 0));

        Mat srcROI;
        //需要计算的图像的通道，灰度图像为0，BGR图像需要指定B,G,R
        const int channels[] = { 0 };
        //    OutputArray hist2 = hist1;
        //Mat hist;//定义输出Mat类型
        Mat hist1;//定义输出Mat类型
        int dims = 1;//设置直方图维度
        const int histSize[] = { 256 }; //直方图每一个维度划分的柱条的数目
        //每一个维度取值范围
        float pranges[] = { 0, 255 };//取值区间
        const float* ranges[] = { pranges };
        vector<double> hist_list;
        vector<Mat> roi_list;

        //Mat roi = imread("D:/roi.png", 0);
        //calcHist(&roi, 1, channels, Mat(), hist1, dims, histSize, ranges, true, false);//计算直方图
        //normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
        //for (int x = 0; x < drawImg.size[1]; x = x + drawImg.size[1] / 4) {

        //    srcROI = destMat(Rect(x, 0, sourceMat_src.cols / 4, sourceMat_src.rows));
        //    calcHist(&srcROI, 1, channels, Mat(), hist, dims, histSize, ranges, true, false);//计算直方图
        //    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
        //    double base_base = compareHist(hist, hist1, 1);
        //    hist_list.push_back(base_base);
        //    roi_list.push_back(srcROI);
        //}
        //int maxPosition = max_element(hist_list.begin(), hist_list.end()) - hist_list.begin();
        //*arr = hist_list[maxPosition];
        for (int i = 0; i < contours.size(); i++) {
            RotatedRect rect = minAreaRect(contours[i]);
            /*if (rect.size.width > 500 || rect.size.height > 500) {
                outfile << rect.size.width << "--" << rect.size.height << endl;
            }*/
            if ((rect.size.width > 1000 || rect.size.height > 1000) && (rect.size.width < 60 || rect.size.height < 60)) {
                if (!center_y[0]) {
                    best_contours.push_back(contours[i]);
                    center_y[0] = rect.center.y;
                    if (rect.size.width < rect.size.height) {
                        rotated = 90 - rect.angle;
                    }
                    else {
                        rotated = rect.angle;
                    }
                }
                else if (!center_y[1]) {
                    if (abs(rect.center.y - center_y[0]) > 150) {
                        best_contours.push_back(contours[i]);
                        center_y[1] = rect.center.y;
                    }
                }
            }
            /*if (minAreaRect(contours[i]).size.width < 50 && minAreaRect(contours[i]).size.height>1000) {
                best_contours.push_back(contours[i]);
                break;
            }*/
            // Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        }
        if (best_contours.size() == 2) {
            if (center_y[0] > center_y[1]) {
                srcROI = destMat(Rect(0, center_y[1] - 10, destMat.cols, abs(center_y[1] - center_y[0]) + 20));
            }
            else {
                srcROI = destMat(Rect(0, center_y[0] - 10, destMat.cols, abs(center_y[1] - center_y[0]) + 20));
            }
            calcHist(&srcROI, 1, channels, Mat(), hist1, dims, histSize, ranges, true, false);//计算直方图
            //int max_val;
            //    minMaxLoc(srcROI, 0, &max_val, 0, 0);//计算直方图的最大像素值
            //    cout<< "maxval" << hist1<< endl;
            arr = hist1.at<float>(0, 0);
            //输入拟合点
            //std::vector<cv::Point> points = best_contours[0];
            cv::Mat A;
            polynomial_curve_fit(best_contours[0], 9, A);
            //std::cout << "A = " << A << drawImg.size[1] << std::endl;
            std::vector<cv::Point> points_fitted;
            std::vector<double> cycle_y;
            for (int x = 0; x < drawImg.size[1]; x++)
            {
                double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
                    A.at<double>(2, 0) * std::pow(x, 2) + A.at<double>(3, 0) * std::pow(x, 3) + A.at<double>(4, 0) * std::pow(x, 4) + A.at<double>(5, 0) * std::pow(x, 5)
                    + A.at<double>(6, 0) * std::pow(x, 6) + A.at<double>(7, 0) * std::pow(x, 7) + A.at<double>(8, 0) * std::pow(x, 8) + A.at<double>(9, 0) * std::pow(x, 9);
                points_fitted.push_back(cv::Point(x, y));
                cycle_y.push_back(y);
            }
            //画线
            //cv::polylines(drawImg, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
            //    cv::imshow("image", image);
            //    cv::polylines(drawImg, points_fitted, false, cv::Scalar(255, 255, 255), 1, 8, 0);

            //double temp[1024];
            //    for(auto i:temp)cout<<i<<endl;
            vector<int> result;
            vector<int> ind = findPeaks(cycle_y, 1024);
            if (ind.size() > 3) {
                for (int i = 1; i < ind.size() - 2; i++) {
                    cout << "index:" << ind[i + 1] - ind[i] << endl;
                    result.push_back(ind[i + 1] - ind[i]);
                };
            };
            double sumValue = accumulate(std::begin(result), std::end(result), 0.0);
            double meanValue = sumValue / result.size();
            auto end = chrono::system_clock::now(); // 结束时间
            *lay = meanValue / cos(a2r(rotated));
            //*angle = rotated;
            *time = getSeconds(start, end);
            //*diameter = abs(center_y[1] - center_y[0]) * cos(a2r(rotated));
            //*swing = abs((center_y[0] + center_y[1]) / 2 - drawImg.size[0] / 2);
            cv::cvtColor(srcROI, srcROI, CV_RGB2BGRA);
        }
        else {
            auto end = chrono::system_clock::now(); // 结束时间
            *lay = 0;
            //*angle = 0;
            *time = getSeconds(start, end);
            //*diameter = 0;
            //*swing = 0;
            cv::cvtColor(destMat, srcROI, CV_RGB2BGRA);
        }


        //outfile << "success3" << endl;

        ThrowNIError(dest.MatToImage(srcROI));

    }
    catch (NIERROR& _err) {
        error = _err;
    }
    catch (std::string e) {
        outfile << e << endl;
        error = NI_ERR_OCV_USER;
    }
    ProcessNIError(error, errorHandle);
}

EXTERN_C void NI_EXPORT detect_diameter(NIImageHandle sourceHandle_src, NIImageHandle destHandle, NIErrorHandle errorHandle, double* diameter, double* time) {
    //auto file_logger = spdlog::basic_logger_mt("basic_logger", "D:/basic.txt");
    //spdlog::set_default_logger(file_logger);
    NIERROR error = NI_ERR_SUCCESS;
    ReturnOnPreviousError(errorHandle);
    ofstream outfile;
    vector<vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;
    vector<vector<cv::Point>> best_contours;
    vector<int> center_y(2);
    float rotated;
    float arr;
    try
    {
        if (!sourceHandle_src || !destHandle || !errorHandle) {
            ThrowNIError(NI_ERR_NULL_POINTER);
        }
        NIImage source_src(sourceHandle_src);
        NIImage dest(destHandle);

        Mat sourceMat_src;
        Mat destMat;
        // ni图片转Mat
        ThrowNIError(source_src.ImageToMat(sourceMat_src));
        // 如果是彩色图片,需要转换

        // 以写模式打开文件

        //outfile.open("D:/afile.txt");


        // 向文件写入用户输入的数据
        //outfile << source_src.type << endl;
        auto start = chrono::system_clock::now(); // 开始时间
        // 文字位置
        Point org(10, 90);
        if (source_src.type == NIImage_RGB32)
        {
            cv::cvtColor(sourceMat_src, sourceMat_src, CV_BGRA2GRAY);
            //outfile << "success" << endl;
            //imwrite("D:/haha1.png", destMat);

        }
        cv::transpose(sourceMat_src, sourceMat_src);
        cv::flip(sourceMat_src, sourceMat_src, -1);

        cv::adaptiveThreshold(sourceMat_src, destMat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, -2);

        Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
        cv::erode(destMat, destMat, element);
        //imwrite("D:/haha2.png", destMat);
        cv::findContours(destMat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        // sort contours
        //std::sort(contours.begin(), contours.end(), compareContourAreas);
        Mat drawImg = Mat::zeros(destMat.size(), CV_8UC3);
        drawImg.setTo(cv::Scalar(0, 0, 0));

        Mat srcROI;
        //需要计算的图像的通道，灰度图像为0，BGR图像需要指定B,G,R
        const int channels[] = { 0 };
        //    OutputArray hist2 = hist1;
        //Mat hist;//定义输出Mat类型
        Mat hist1;//定义输出Mat类型
        int dims = 1;//设置直方图维度
        const int histSize[] = { 256 }; //直方图每一个维度划分的柱条的数目
        //每一个维度取值范围
        float pranges[] = { 0, 255 };//取值区间
        const float* ranges[] = { pranges };
        vector<double> hist_list;
        vector<Mat> roi_list;

        //Mat roi = imread("D:/roi.png", 0);
        //calcHist(&roi, 1, channels, Mat(), hist1, dims, histSize, ranges, true, false);//计算直方图
        //normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
        //for (int x = 0; x < drawImg.size[1]; x = x + drawImg.size[1] / 4) {

        //    srcROI = destMat(Rect(x, 0, sourceMat_src.cols / 4, sourceMat_src.rows));
        //    calcHist(&srcROI, 1, channels, Mat(), hist, dims, histSize, ranges, true, false);//计算直方图
        //    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
        //    double base_base = compareHist(hist, hist1, 1);
        //    hist_list.push_back(base_base);
        //    roi_list.push_back(srcROI);
        //}
        //int maxPosition = max_element(hist_list.begin(), hist_list.end()) - hist_list.begin();
        //*arr = hist_list[maxPosition];
        for (int i = 0; i < contours.size(); i++) {
            RotatedRect rect = minAreaRect(contours[i]);
            /*if (rect.size.width > 500 || rect.size.height > 500) {
                outfile << rect.size.width << "--" << rect.size.height << endl;
            }*/
            if ((rect.size.width > 1000 || rect.size.height > 1000) && (rect.size.width < 60 || rect.size.height < 60)) {
                if (!center_y[0]) {
                    best_contours.push_back(contours[i]);
                    center_y[0] = rect.center.y;
                    if (rect.size.width < rect.size.height) {
                        rotated = 90 - rect.angle;
                    }
                    else {
                        rotated = rect.angle;
                    }
                }
                else if (!center_y[1]) {
                    if (abs(rect.center.y - center_y[0]) > 150) {
                        best_contours.push_back(contours[i]);
                        center_y[1] = rect.center.y;
                    }
                }
            }
            /*if (minAreaRect(contours[i]).size.width < 50 && minAreaRect(contours[i]).size.height>1000) {
                best_contours.push_back(contours[i]);
                break;
            }*/
            // Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        }
        if (best_contours.size() == 2) {
            /*if (center_y[0] > center_y[1]) {
                srcROI = destMat(Rect(0, center_y[1] - 10, destMat.cols, abs(center_y[1] - center_y[0]) + 20));
            }
            else {
                srcROI = destMat(Rect(0, center_y[0] - 10, destMat.cols, abs(center_y[1] - center_y[0]) + 20));
            }*/
            //calcHist(&srcROI, 1, channels, Mat(), hist1, dims, histSize, ranges, true, false);//计算直方图
            //int max_val;
            //    minMaxLoc(srcROI, 0, &max_val, 0, 0);//计算直方图的最大像素值
            //    cout<< "maxval" << hist1<< endl;
            //arr = hist1.at<float>(0, 0);
            ////输入拟合点
            ////std::vector<cv::Point> points = best_contours[0];
            //cv::Mat A;
            //polynomial_curve_fit(best_contours[0], 9, A);
            ////std::cout << "A = " << A << drawImg.size[1] << std::endl;
            //std::vector<cv::Point> points_fitted;
            //std::vector<double> cycle_y;
            /*for (int x = 0; x < drawImg.size[1]; x++)
            {
                double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
                    A.at<double>(2, 0) * std::pow(x, 2) + A.at<double>(3, 0) * std::pow(x, 3) + A.at<double>(4, 0) * std::pow(x, 4) + A.at<double>(5, 0) * std::pow(x, 5)
                    + A.at<double>(6, 0) * std::pow(x, 6) + A.at<double>(7, 0) * std::pow(x, 7) + A.at<double>(8, 0) * std::pow(x, 8) + A.at<double>(9, 0) * std::pow(x, 9);
                points_fitted.push_back(cv::Point(x, y));
                cycle_y.push_back(y);
            }*/
            //画线
            //cv::polylines(drawImg, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
            //    cv::imshow("image", image);
            //    cv::polylines(drawImg, points_fitted, false, cv::Scalar(255, 255, 255), 1, 8, 0);

            //double temp[1024];
            //    for(auto i:temp)cout<<i<<endl;
            /*vector<int> result;
            vector<int> ind = findPeaks(cycle_y, 1024);
            if (ind.size() > 3) {
                for (int i = 1; i < ind.size() - 2; i++) {
                    cout << "index:" << ind[i + 1] - ind[i] << endl;
                    result.push_back(ind[i + 1] - ind[i]);
                };
            };
            double sumValue = accumulate(std::begin(result), std::end(result), 0.0);
            double meanValue = sumValue / result.size();*/
            auto end = chrono::system_clock::now(); // 结束时间
            //*lay = meanValue / cos(a2r(rotated));
            //*angle = rotated;
            *time = getSeconds(start, end);
            *diameter = abs(center_y[1] - center_y[0]) * cos(a2r(rotated));
            //*swing = abs((center_y[0] + center_y[1]) / 2 - drawImg.size[0] / 2);
            cv::cvtColor(destMat, srcROI, CV_RGB2BGRA);
        }
        else {
            auto end = chrono::system_clock::now(); // 结束时间
            //*lay = 0;
            //*angle = 0;
            *time = getSeconds(start, end);
            *diameter = 0;
            //*swing = 0;
            cv::cvtColor(destMat, srcROI, CV_RGB2BGRA);
        }


        //outfile << "success3" << endl;

        ThrowNIError(dest.MatToImage(srcROI));

    }
    catch (NIERROR& _err) {
        error = _err;
    }
    catch (std::string e) {
        outfile << e << endl;
        error = NI_ERR_OCV_USER;
    }
    ProcessNIError(error, errorHandle);
}

EXTERN_C void NI_EXPORT detect_swing(NIImageHandle sourceHandle_src, NIImageHandle destHandle, NIErrorHandle errorHandle, float* angle, double* time, double* swing) {
    //auto file_logger = spdlog::basic_logger_mt("basic_logger", "D:/basic.txt");
    //spdlog::set_default_logger(file_logger);
    NIERROR error = NI_ERR_SUCCESS;
    ReturnOnPreviousError(errorHandle);
    ofstream outfile;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<vector<Point>> best_contours;
    vector<int> center_y(2);
    float rotated;
    float arr;
    try
    {
        if (!sourceHandle_src || !destHandle || !errorHandle) {
            ThrowNIError(NI_ERR_NULL_POINTER);
        }
        NIImage source_src(sourceHandle_src);
        NIImage dest(destHandle);

        Mat sourceMat_src;
        Mat destMat;
        // ni图片转Mat
        ThrowNIError(source_src.ImageToMat(sourceMat_src));
        // 如果是彩色图片,需要转换
        //spdlog::info("Welcome to spdlog!");
        //ilog->info("ss:{}",source_src.type);
        // 以写模式打开文件
        //outfile.open("D:/afile.txt");
        outfile << source_src.type << endl;
        auto start = chrono::system_clock::now(); // 开始时间
        // 文字位置
        Point org(10, 90);
        if (source_src.type == NIImage_RGB32)
        {
            cv::cvtColor(sourceMat_src, sourceMat_src, CV_BGRA2GRAY);
            //outfile << "success" << endl;
            //imwrite("D:/haha1.png", destMat);

        }
        cv::transpose(sourceMat_src, sourceMat_src);
        cv::flip(sourceMat_src, sourceMat_src, -1);

        cv::adaptiveThreshold(sourceMat_src, destMat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, -2);

        Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
        cv::erode(destMat, destMat, element);
        //imwrite("D:/haha2.png", destMat);
        cv::findContours(destMat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        // sort contours
        //std::sort(contours.begin(), contours.end(), compareContourAreas);
        Mat drawImg = Mat::zeros(destMat.size(), CV_8UC3);
        drawImg.setTo(cv::Scalar(0, 0, 0));

        Mat srcROI;
        //需要计算的图像的通道，灰度图像为0，BGR图像需要指定B,G,R
        const int channels[] = { 0 };
        //    OutputArray hist2 = hist1;
        //Mat hist;//定义输出Mat类型
        Mat hist1;//定义输出Mat类型
        int dims = 1;//设置直方图维度
        const int histSize[] = { 256 }; //直方图每一个维度划分的柱条的数目
        //每一个维度取值范围
        float pranges[] = { 0, 255 };//取值区间
        const float* ranges[] = { pranges };
        vector<double> hist_list;
        vector<Mat> roi_list;

        for (int i = 0; i < contours.size(); i++) {
            RotatedRect rect = minAreaRect(contours[i]);
            /*if (rect.size.width > 500 || rect.size.height > 500) {
                outfile << rect.size.width << "--" << rect.size.height << endl;
            }*/
            if ((rect.size.width > 1000 || rect.size.height > 1000) && (rect.size.width < 60 || rect.size.height < 60)) {
                if (!center_y[0]) {
                    best_contours.push_back(contours[i]);
                    center_y[0] = rect.center.y;
                    if (rect.size.width < rect.size.height) {
                        rotated = 90 - rect.angle;
                    }
                    else {
                        rotated = rect.angle;
                    }
                }
                else if (!center_y[1]) {
                    if (abs(rect.center.y - center_y[0]) > 150) {
                        best_contours.push_back(contours[i]);
                        center_y[1] = rect.center.y;
                    }
                }
            }
        }
        if (best_contours.size() == 2) {
            if (center_y[0] > center_y[1]) {
                srcROI = destMat(Rect(0, center_y[1] - 10, destMat.cols, abs(center_y[1] - center_y[0]) + 20));
            }
            else {
                srcROI = destMat(Rect(0, center_y[0] - 10, destMat.cols, abs(center_y[1] - center_y[0]) + 20));
            }
            calcHist(&srcROI, 1, channels, Mat(), hist1, dims, histSize, ranges, true, false);//计算直方图
            arr = hist1.at<float>(0, 0);
            //输入拟合点
            //std::vector<cv::Point> points = best_contours[0];
            cv::Mat A;
            polynomial_curve_fit(best_contours[0], 9, A);
            //std::cout << "A = " << A << drawImg.size[1] << std::endl;
            std::vector<cv::Point> points_fitted;
            std::vector<double> cycle_y;
            for (int x = 0; x < drawImg.size[1]; x++)
            {
                double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
                    A.at<double>(2, 0) * std::pow(x, 2) + A.at<double>(3, 0) * std::pow(x, 3) + A.at<double>(4, 0) * std::pow(x, 4) + A.at<double>(5, 0) * std::pow(x, 5)
                    + A.at<double>(6, 0) * std::pow(x, 6) + A.at<double>(7, 0) * std::pow(x, 7) + A.at<double>(8, 0) * std::pow(x, 8) + A.at<double>(9, 0) * std::pow(x, 9);
                points_fitted.push_back(cv::Point(x, y));
                cycle_y.push_back(y);
            }
            vector<int> result;
            vector<int> ind = findPeaks(cycle_y, 1024);
            if (ind.size() > 3) {
                for (int i = 1; i < ind.size() - 2; i++) {
                    cout << "index:" << ind[i + 1] - ind[i] << endl;
                    result.push_back(ind[i + 1] - ind[i]);
                };
            };
            double sumValue = accumulate(std::begin(result), std::end(result), 0.0);
            double meanValue = sumValue / result.size();
            auto end = chrono::system_clock::now(); // 结束时间
            //*lay = meanValue / cos(a2r(rotated));
            *angle = rotated;
            *time = getSeconds(start, end);
            //*diameter = abs(center_y[1] - center_y[0]) * cos(a2r(rotated));
            *swing = abs((center_y[0] + center_y[1]) / 2 - drawImg.size[0] / 2);
            cv::cvtColor(srcROI, srcROI, CV_RGB2BGRA);
        }
        else {
            auto end = chrono::system_clock::now(); // 结束时间
            //*lay = 0;
            *angle = 0;
            *time = getSeconds(start, end);
            //*diameter = 0;
            *swing = 0;
            cv::cvtColor(destMat, srcROI, CV_RGB2BGRA);
        }


        //outfile << "success3" << endl;

        ThrowNIError(dest.MatToImage(srcROI));

    }
    catch (NIERROR& _err) {
        error = _err;
    }
    catch (std::string e) {
        //outfile << e << endl;
        error = NI_ERR_OCV_USER;
    }
    ProcessNIError(error, errorHandle);
}

EXTERN_C void NI_EXPORT detect_all(NIImageHandle sourceHandle_src,NIImageHandle destHandle,NIErrorHandle errorHandle,double* lay,float* angle,double* diameter,double* time,double* swing,float* arr){
    //auto file_logger = spdlog::basic_logger_mt("basic_logger", "D:/basic.txt");
    //spdlog::set_default_logger(file_logger);
    NIERROR error = NI_ERR_SUCCESS;
    ReturnOnPreviousError(errorHandle);
    ofstream outfile;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<vector<Point>> best_contours;
    vector<int> center_y(2);
    float rotated;
    vector<Point> edge1;
    vector<Point> edge2;
    try
    {
        if (!sourceHandle_src || !destHandle || !errorHandle) {
            ThrowNIError(NI_ERR_NULL_POINTER);
        }
        NIImage source_src(sourceHandle_src);
        NIImage dest(destHandle);

        Mat sourceMat_src;
        Mat destMat;
        // ni图片转Mat
        ThrowNIError(source_src.ImageToMat(sourceMat_src));
        // 如果是彩色图片,需要转换
        //spdlog::info("Welcome to spdlog!");
        //ilog->info("ss:{}",source_src.type);
        // 以写模式打开文件
        //outfile.open("D:/afile.txt");
        //outfile << source_src.type << endl;
        auto start = chrono::system_clock::now(); // 开始时间
        // 文字位置
        Point org(10, 90);
        if (source_src.type == NIImage_RGB32)
        {
            cv::cvtColor(sourceMat_src, sourceMat_src, CV_BGRA2GRAY);
            //outfile << "success" << endl;
            //imwrite("D:/haha1.png", destMat);

        }
        cv::transpose(sourceMat_src, sourceMat_src);
        cv::flip(sourceMat_src, sourceMat_src, -1);

        cv::adaptiveThreshold(sourceMat_src, destMat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, -2);

        //腐蚀操作
        Mat conv_kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        Mat erode_kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        //    Mat dstImage;
        erode(destMat, destMat, erode_kernel);//腐蚀
        dilate(destMat, destMat, conv_kernel);//膨胀

        // 寻找边界点操作
        int nchannels = destMat.channels();
        if (1 == nchannels)
        {
            for (int j = 0; j < destMat.size[1]; j++)
                // 先遍历列再遍历行
            {
                for (int i = 0; i < destMat.size[0]; i++)
                {
                    if (destMat.at<uchar>(i, j) == 255) {
                        edge1.push_back(Point(j, i));
                        //                    cout<< i << j <<endl;
                        break;
                    }
                }
                for (int i = destMat.size[0]; i > 0; i--)
                {
                    if (destMat.at<uchar>(i - 1, j) == 255) {
                        edge2.push_back(Point(j, i - 1));
                        //                    cout<< i << j <<endl;
                        break;
                    }
                    //                destMat.at<uchar>(i,j) = 255-destMat.at<uchar>(i,j);
                }
            }
        }


        // sort contours
        //std::sort(contours.begin(), contours.end(), compareContourAreas);
        Mat drawImg = Mat::zeros(destMat.size(), CV_8UC3);
        drawImg.setTo(cv::Scalar(0, 0, 0));

        Mat srcROI;
        //需要计算的图像的通道，灰度图像为0，BGR图像需要指定B,G,R
        const int channels[] = { 0 };
        //    OutputArray hist2 = hist1;
        //Mat hist;//定义输出Mat类型
        Mat hist1;//定义输出Mat类型
        int dims = 1;//设置直方图维度
        const int histSize[] = { 256 }; //直方图每一个维度划分的柱条的数目
        //每一个维度取值范围
        float pranges[] = { 0, 255 };//取值区间
        const float* ranges[] = { pranges };
        vector<double> hist_list;
        vector<Mat> roi_list;
        
        //设置边缘值
        double mean1 = 0;
        double mean2 = 0;
        best_contours.push_back(Mad(edge1, mean1));
        best_contours.push_back(Mad(edge2, mean2));
        center_y[0] = mean1;
        center_y[1] = mean2;

        //outfile << best_contours.size() << endl;
        if (best_contours.size() == 2) {
            if (center_y[0] > center_y[1]) {
                srcROI = destMat(Rect(0, center_y[1] - 10, destMat.cols, abs(center_y[1] - center_y[0]) + 20));
            }
            else {
                srcROI = destMat(Rect(0, center_y[0] - 10, destMat.cols, abs(center_y[1] - center_y[0]) + 20));
            }
            calcHist(&srcROI, 1, channels, Mat(), hist1, dims, histSize, ranges, true, false);//计算直方图
            *arr = hist1.at<float>(0, 0);
            //输入拟合点
            //std::vector<cv::Point> points = best_contours[0];
            cv::Mat A;
            polynomial_curve_fit(best_contours[0], 9, A);
            //std::cout << "A = " << A << drawImg.size[1] << std::endl;
            std::vector<cv::Point> points_fitted;
            std::vector<double> cycle_y;
            for (int x = 0; x < drawImg.size[1]; x++)
            {
                double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
                    A.at<double>(2, 0) * std::pow(x, 2) + A.at<double>(3, 0) * std::pow(x, 3) + A.at<double>(4, 0) * std::pow(x, 4) + A.at<double>(5, 0) * std::pow(x, 5)
                    + A.at<double>(6, 0) * std::pow(x, 6) + A.at<double>(7, 0) * std::pow(x, 7) + A.at<double>(8, 0) * std::pow(x, 8) + A.at<double>(9, 0) * std::pow(x, 9);
                points_fitted.push_back(cv::Point(x, y));
                cycle_y.push_back(y);
            }
            vector<int> result;
            vector<int> ind = findPeaks(cycle_y, 1024);

            if (ind.size() > 2) {
                for (int i = 1; i < ind.size() - 1; i++) {
                    result.push_back(ind[i + 1] - ind[i]);
                }
            }
            double sumValue = accumulate(std::begin(result), std::end(result), 0.0);
            double meanValue = sumValue / result.size();
            auto end = chrono::system_clock::now(); // 结束时间
            *lay = meanValue;
            *angle = 0;
            *time = getSeconds(start, end);
            *diameter = abs(center_y[1] - center_y[0]);
            *swing = abs((center_y[0] + center_y[1]) / 2 - drawImg.size[0] / 2);
            cv::cvtColor(srcROI, srcROI, CV_RGB2BGRA);
        }
        else {
            auto end = chrono::system_clock::now(); // 结束时间
            *lay = 0;
            *angle = 0;
            *time = getSeconds(start, end);
            *diameter = 0;
            *swing = 0;
            cv::cvtColor(destMat, srcROI, CV_RGB2BGRA);

        }


        //outfile << "success3" << endl;

        ThrowNIError(dest.MatToImage(srcROI));

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