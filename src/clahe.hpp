#pragma once

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using timer = std::chrono::steady_clock;

class CLAHE
{
public:
	CLAHE();
	~CLAHE();

	void apply(const cv::Mat &input, const int tile_size, float clip_threshold);

private:
	int m_height;
	int m_width;
	float m_clip_threshold;
};