#include "clahe.hpp"

CLAHE::CLAHE() {}

CLAHE::~CLAHE() {}

static std::vector<float> histogram(cv::Mat &channel)
{
	CV_Assert(channel.type() == CV_32F);

	cv::Mat hist;
	const int hist_size = 256;
	float range[] = {0.0f, 1.0f};
	const float *histRange = {range};

	cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &hist_size, &histRange);

	// normalize to PDF
	hist /= channel.total();

	// CDF
	std::vector<float> lut(hist_size);
	lut[0] = 0.0f;
	for (int i = 1; i < hist_size; ++i)
	{
		hist.at<float>(i) += hist.at<float>(i - 1);
		lut[i] = hist.at<float>(i);
	}

	// overwrite channel
	return lut;
}

static float lineer_interpolation(float dist_x1, float dist_x2, float x1_val, float x2_val)
{
	float sum = dist_x1 + dist_x2;
	float interpolated_value = (dist_x1 / sum) * x2_val + (dist_x2 / sum) * x1_val;

	return interpolated_value;
}

cv::Mat CLAHE::apply(const cv::Mat &input, const int tile_size, float clip_threshold)
{
	m_height = input.rows;
	m_width = input.cols;

	// normalize & conversion to lab
	cv::Mat input_normalized, input_lab;
	input.convertTo(input_normalized, CV_32F, 1.0f / 255.0f);
	cv::cvtColor(input_normalized, input_lab, cv::COLOR_BGR2Lab);

	// get lab channels
	std::vector<cv::Mat> channels;
	cv::split(input_lab, channels);

	const int tile_num = m_height / tile_size;

	std::vector<std::vector<float>> tile_luts(tile_num * tile_num);

	// get transformation luts for each tile
	for (int i = 0; i < tile_num; i++)
	{
		for (int j = 0; j < tile_num; j++)
		{
			cv::Rect roi(j * tile_size, i * tile_size, tile_size, tile_size);

			cv::Mat v_tile = channels[0](roi);
			tile_luts[i * tile_num + j] = histogram(v_tile);
		}
	}

	for (int i = tile_size / 2; i < m_height - tile_size / 2; i++)
	{
		for (int j = tile_size / 2; j < m_width - tile_size / 2; j++)
		{
			int v = std::min(255, static_cast<int>(channels[0].at<float>(i, j) * 255.0f));

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

			channels[0].at<float>(i, j) = vlast;
		}
	}

	cv::merge(channels, input_lab);

	// re-convert to bgr & de-normalize
	cv::Mat output_bgr;
	cv::cvtColor(input_lab, output_bgr, cv::COLOR_Lab2BGR);
	output_bgr.convertTo(output_bgr, CV_8U, 255.0);

	return output_bgr;
}
