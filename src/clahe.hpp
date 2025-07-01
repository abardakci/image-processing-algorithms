#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

class CLAHE
{
public:
	CLAHE();
	~CLAHE();

	cv::Mat apply(const cv::Mat &input, const int tile_size, float clip_threshold);

private:
	int m_height;
	int m_width;
};