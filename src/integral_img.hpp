#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

void convert_to_integral_image(cv::Mat &input)
{
	cv::Mat output;

	int w = input.cols;
	int h = input.rows;

	float *ptr = input.ptr<float>();

	for (int j = 1; j < w; ++j)
	{
		ptr[j] += ptr[j - 1];
	}

	for (int i = 1; i < h; ++i)
	{
		ptr[i * w] += ptr[(i - 1) * w];

		for (int j = 1; j < w; ++j)
		{
			ptr[i * w + j] += ptr[(i - 1) * w + j] + ptr[i * w + j - 1] - ptr[(i - 1) * w + (j - 1)];
		}
	}
}
