#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>
#include <algorithm>
#include <vector>

// x: col
// y: row
class xylab : public cv::Vec<float, 5>
{
public:
    // Constructors
    xylab(float val) : cv::Vec<float, 5>()
    {
        (*this)[0] = val;
        (*this)[1] = val;
        (*this)[2] = val;
        (*this)[3] = val;
        (*this)[4] = val;
    }

    xylab(float x, float y, float l, float a, float b)
    {
        (*this)[0] = x;
        (*this)[1] = y;
        (*this)[2] = l;
        (*this)[3] = a;
        (*this)[4] = b;
    }

    // Getters
    float x() const { return (*this)[0]; }
    float y() const { return (*this)[1]; }
    float L() const { return (*this)[2]; }
    float A() const { return (*this)[3]; }
    float B() const { return (*this)[4]; }

    // Setters
    void setX(float val) { (*this)[0] = val; }
    void setY(float val) { (*this)[1] = val; }
    void setL(float val) { (*this)[2] = val; }
    void setA(float val) { (*this)[3] = val; }
    void setB(float val) { (*this)[4] = val; }
};

class Centroid
{
public:
    Centroid() : xylab_(0.0f), label_(-1) {}
    Centroid(xylab m, int id) : xylab_(m), label_(id) {}

    xylab xylab_;
    int label_;
};

class PixelCtx
{
public:
    PixelCtx(int size);

    std::vector<int> labels_;
    std::vector<float> distances_;
};

cv::Mat SLIC(const cv::Mat &input, int num_sp, float T, const int kMaxIterNum);
