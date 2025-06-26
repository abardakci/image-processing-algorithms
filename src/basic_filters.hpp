#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

cv::Mat histogramEqualization(cv::Mat& input);
cv::Mat histogramEqualizationV2(cv::Mat& input);
cv::Mat boxFilter(cv::Mat& input, const int filter_size);
