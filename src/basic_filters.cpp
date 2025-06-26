#include "basic_filters.hpp"

int clamp(int val, int low, int high) 
{
    return std::max(low, std::min(val, high));
}

cv::Mat histogramEqualization(cv::Mat& input)
{
    CV_Assert(input.type() == CV_8UC1); // Gri seviye kontrolü

    cv::Mat hist;
    const int hist_size = 256;        // 0-255
    float range[] = { 0, 256 }; // 256 dış sınır
    const float* histRange = { range };

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

cv::Mat boxFilter(cv::Mat& input, const int filter_size)
{
    int flow = filter_size / 2;  

    int h = input.rows;
    int w = input.cols;
    int hw = h * w;

    cv::Mat dst(cv::Size(w,h), input.type());

    int filter_area = filter_size * filter_size;

    // fast path
    for (int i = flow; i < h - flow; ++i)
    {
        for (int j = flow; j < w - flow; ++j)
        {
            float accumulator = 0.0f;
            for (int m = 0; m < filter_area; ++m)
            {
                int row = m / 3;
                int col = m % 3;

                accumulator += input.at<uchar>(i - flow + row, j - flow + col);
            }

            dst.at<uchar>(i, j) = static_cast<uchar>(accumulator / filter_area);
        }
    }

    // Slow path: kenar bölgeler
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            bool i_con = i >= flow && i < h-flow;
            bool j_con = j == flow;
            if (i_con && j_con)
            {
                j = w - flow;
                continue; // fast path'te zaten yapıldı
            }
            float accumulator = 0.0f;
            for (int m = -flow; m <= flow; ++m)
            {
                int y = clamp(i + m, 0, h - 1);
                for (int n = -flow; n <= flow; ++n)
                {
                    int x = clamp(j + n, 0, w - 1);
                    accumulator += input.at<uchar>(y, x);
                }
            }

            dst.at<uchar>(i, j) = static_cast<uchar>(accumulator / filter_area);
        }
    }

    return dst;
}

cv::Mat boxFilterV2(cv::Mat& input, const int filter_size)
{
    int overflow = filter_size / 2;  

    int h = input.rows;
    int w = input.cols;

    cv::Mat dst(cv::Size(w,h), input.type());

    int filter_area = filter_size * 1;
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            int offset_i = i - overflow;
            for (int m = 0; m < filter_size; ++m)
            {
                input.at<uchar>(offset_i + m, j);
            }   

            dst.at<uchar>(i, j) = -1;
        }
    }

    return dst;
}

cv::Mat medianFilter(cv::Mat& input, const int filter_size)
{
    return input;
}
