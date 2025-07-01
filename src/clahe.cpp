#include "clahe.hpp"

CLAHE::CLAHE() {}

CLAHE::~CLAHE() {}

static void histogram(cv::Mat &v_channel)
{
	CV_Assert(v_channel.type() == CV_32F);

	cv::Mat hist;
	const int hist_size = 256;
	float range[] = {0.0f, 1.0f};
	const float *histRange = {range};

	cv::calcHist(&v_channel, 1, 0, cv::Mat(), hist, 1, &hist_size, &histRange);

	// normalize to PDF
	hist /= v_channel.total();

	// CDF hesapla
	for (int i = 1; i < hist_size; ++i)
	{
		hist.at<float>(i) += hist.at<float>(i - 1);
	}

	// CDF LUT
	std::vector<float> lut(hist_size);
	for (int i = 0; i < hist_size; ++i)
	{
		lut[i] = hist.at<float>(i); // ∈ [0, 1]
	}

	// eşitleme uygula
	for (int i = 0; i < v_channel.rows; ++i)
	{
		for (int j = 0; j < v_channel.cols; ++j)
		{
			float val = v_channel.at<float>(i, j);
			int bin = std::min(int(val * 255.0f), 255);
			v_channel.at<float>(i, j) = lut[bin];
		}
	}
}

cv::Mat CLAHE::apply(const cv::Mat &input, const int tile_size, float clip_threshold)
{
	m_height = input.rows;
	m_width = input.cols;

	// 1. Normalize and convert to HSV
	cv::Mat input_normalized, input_hsv;
	input.convertTo(input_normalized, CV_32F, 1.0f / 255.0f);
	cv::cvtColor(input_normalized, input_hsv, cv::COLOR_BGR2HSV);

	// 2. HSV kanallarına ayır
	std::vector<cv::Mat> hsv_channels;
	cv::split(input_hsv, hsv_channels); // hsv_channels[2] = V

	// 3. Tile tile V kanalına eşitleme uygula
	for (int i = 0; i < m_height; i += tile_size)
	{
		for (int j = 0; j < m_width; j += tile_size)
		{
			cv::Rect roi(j, i, tile_size, tile_size);

			cv::Mat v_tile = hsv_channels[2](roi);
			histogram(v_tile); // doğrudan değiştirme
		}
	}

	// 4. HSV'yi yeniden birleştirip BGR’ye dön
	cv::merge(hsv_channels, input_hsv);

	cv::Mat output_bgr;
	cv::cvtColor(input_hsv, output_bgr, cv::COLOR_HSV2BGR);
	output_bgr.convertTo(output_bgr, CV_8U, 255.0);

	return output_bgr;
}
