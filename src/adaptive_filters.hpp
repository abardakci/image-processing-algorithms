#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

cv::Mat adaptive_mean_filter(cv::Mat &input, int filter_size, float noise_variance);

cv::Mat adaptive_median_filter(cv::Mat &input, int max_filter_size);