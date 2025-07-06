#pragma once

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

class NLM
{
public:
	NLM();
	~NLM();

	cv::Mat apply(const cv::Mat &input, const int patch_size, const int window_size, const float h);

private:
	int m_height;
	int m_width;

};