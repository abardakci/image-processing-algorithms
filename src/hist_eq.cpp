#include "hist_eq.hpp"

cv::Mat histogramEqualization(cv::Mat &input)
{
    CV_Assert(input.type() == CV_8UC1); // Gri seviye kontrolü

    cv::Mat hist;
    const int hist_size = 256; // 0-255
    float range[] = {0, 256};  // 256 dış sınır
    const float *histRange = {range};

    cv::calcHist(&input, 1, 0, cv::Mat(), hist, 1, &hist_size, &histRange);

    int h = input.rows;
    int w = input.cols;
    int hw = h * w;
    hist /= hw;

    for (int i = 1; i < hist_size; ++i)
    {
        hist.at<float>(i) += hist.at<float>(i - 1);
    }

    cv::Mat lut(1, hist_size, CV_8U);
    for (int i = 0; i < hist_size; ++i)
    {
        lut.at<uchar>(i) = static_cast<uchar>(hist.at<float>(i) * 255.0f);
    }

    cv::Mat output(cv::Size(w, h), CV_8UC1);

    // cv::LUT(input, lut, output);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            int intensity = static_cast<uchar>(input.at<uchar>(i, j));
            output.at<uchar>(i, j) = lut.at<uchar>(intensity);
        }
    }

    return output;
}