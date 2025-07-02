#include "clahe.hpp"

CLAHE::CLAHE() {}

CLAHE::~CLAHE() {}

static std::vector<int> calcHist(cv::Mat& channel, float begin, float end, int bin_size)
{
    CV_Assert(channel.type() == CV_32F);

    std::vector<int> hist(bin_size, 0);
    float step = (end - begin) / bin_size;

    int total = channel.rows * channel.cols;
    const float* data = (float*)channel.data;

    for (int i = 0; i < total; ++i)
    {
        float val = data[i];
        int idx = static_cast<int>((val - begin) / step);
        if (idx >= 0 && idx < bin_size)
        {
            ++hist[idx];
        }
    }

    return hist;
}

static std::vector<float> histogram(cv::Mat &channel, float clip_threshold)
{
	CV_Assert(channel.type() == CV_32F);

    const int hist_size = 256;

	std::vector<int> hist_int = calcHist(channel, 0.0f, 1.0f, 256);

	std::vector<float> pdf(hist_size);
	
	// normalize to PDF
	for (int i = 0; i < hist_size; ++i)
	{
		pdf[i] = (float)hist_int[i] / (float)channel.total(); 
	}

    // for (int i = 0; i < pdf.size(); ++i)
    // {
    //     float &h = pdf[i];
    //     if (h > clip_threshold)
    //     {
	// 		h = clip_threshold;      
    //     }
    // }

	// CDF
	std::vector<float> lut(hist_size);
	lut[0] = pdf[0];
	for (int i = 1; i < hist_size; ++i)
	{
		pdf[i] += pdf[i - 1];
		lut[i] = pdf[i];
	}

	return lut;
}

static float lineer_interpolation(float dist_x1, float dist_x2, float x1_val, float x2_val)
{
	float sum = dist_x1 + dist_x2;
	float interpolated_value = (dist_x1 / sum) * x2_val + (dist_x2 / sum) * x1_val;

	return interpolated_value;
}

static inline float clamp(float input, float min, float max)
{
	if (input < min)
		return min;

	if (input > max)
		return max;

	return input;
}

cv::Mat CLAHE::apply(const cv::Mat &input, const int tile_size, float clip_threshold)
{
	m_height = input.rows;
	m_width = input.cols;
	m_clip_threshold = clip_threshold;

	cv::Mat input_flt, input_hsv;
	input.convertTo(input_flt, CV_32F, 1.0f / 255.0f);
	cv::cvtColor(input_flt, input_hsv, cv::COLOR_BGR2HSV);

	// get hsv channels
	std::vector<cv::Mat> channels;
	cv::split(input_hsv, channels);

	const int tile_num = m_height / tile_size;

	std::vector<std::vector<float>> tile_luts(tile_num * tile_num);

	// get transformation luts for each tile
	for (int i = 0; i < tile_num; i++)
	{
		for (int j = 0; j < tile_num; j++)
		{
			cv::Rect roi(j * tile_size, i * tile_size, tile_size, tile_size);

			cv::Mat v_tile = channels[2](roi);
			tile_luts[i * tile_num + j] = histogram(v_tile, m_clip_threshold);
		}
	}

	for (int i = tile_size / 2; i < m_height - tile_size / 2; i++)
	{
		for (int j = tile_size / 2; j < m_width - tile_size / 2; j++)
		{
			int v = std::min(255, static_cast<int>(channels[2].at<float>(i, j) * 255.0f));

			int i_tile = (i - (tile_size / 2)) / tile_size;
			int j_tile = (j - (tile_size / 2)) / tile_size;

			int di_up = i - (i_tile * tile_size + tile_size / 2);
			int di_bottom = tile_size - di_up;
			int dj_left = j - (j_tile * tile_size + tile_size / 2);
			int dj_right = tile_size - dj_left;

			float c1 = tile_luts[i_tile * tile_num + j_tile][v];
			float c2 = tile_luts[i_tile * tile_num + j_tile + 1][v];
			float c3 = tile_luts[(i_tile + 1) * tile_num + j_tile][v];
			float c4 = tile_luts[(i_tile + 1) * tile_num + j_tile + 1][v];

			float v1 = lineer_interpolation(dj_left, dj_right, c1, c2);
			float v2 = lineer_interpolation(dj_left, dj_right, c3, c4);
			float vlast = lineer_interpolation(di_up, di_bottom, v1, v2);

			channels[2].at<float>(i, j) = clamp(vlast, 0.0f, 1.0f);		
		}
	}

	cv::merge(channels, input_hsv);

	// re-convert to bgr
	cv::Mat output_bgr, output_u8;
	cv::cvtColor(input_hsv, output_bgr, cv::COLOR_HSV2BGR);
	output_bgr.convertTo(output_u8, CV_8U, 255);

	return output_u8;
}
