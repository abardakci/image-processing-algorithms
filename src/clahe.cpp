#include "clahe.hpp"

// fix getting only square images
// add histogram clipping
// cover edges & corners

CLAHE::CLAHE() {}

CLAHE::~CLAHE() {}

static std::vector<int> calcHist(cv::Mat &channel, float begin, float end, int bin_size)
{
	CV_Assert(channel.type() == CV_32F);

	std::vector<int> hist(bin_size, 0);
	float step = (end - begin) / bin_size;

	int total = channel.rows * channel.cols;
	const float *data = (float *)channel.data;

	for (int i = 0; i < total; ++i)
	{
		float val = data[i];
		int idx = static_cast<int>((val - begin) / step);
		assert(idx >= 0 && idx < bin_size);
		++hist[idx];
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
	int total = channel.total();
	for (int i = 0; i < hist_size; ++i)
	{
		pdf[i] = (float)hist_int[i] / total;
	}

	float acc = 0.0f;
	for (int i = 0; i < pdf.size(); ++i)
	{
		float &h = pdf[i];
		if (h > clip_threshold)
		{
			acc += h - clip_threshold;
			h = clip_threshold;
		}
	}

	float dist = acc / pdf.size();
	for (int i = 0; i < pdf.size(); ++i)
	{
		pdf[i] += dist;
	}

	float sum = 0.0f;
	for (float f : pdf)
	{
		sum += f;
	}

	for (float& f: pdf)
	{
		f /= sum;
	}

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
	return (dist_x1 / sum) * x2_val + (dist_x2 / sum) * x1_val;
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

	int wpad = tile_size - (m_width % tile_size);
	int left = wpad / 2;
	int right = wpad - left;

	int hpad = tile_size - (m_height % tile_size);
	int top = hpad / 2;
	int down = hpad - top;

	cv::Mat input_final(cv::Size(m_width + wpad, m_height + hpad), input_hsv.type());
	cv::copyMakeBorder(input_hsv, input_final, top, down, left, right, CV_HAL_BORDER_REFLECT_101);

	// get hsv channels
	std::vector<cv::Mat> channels;
	cv::split(input_final, channels);

	auto t1 = timer::now();
	
	const int wtile_num = input_final.cols / tile_size;
	const int htile_num = input_final.rows / tile_size;

	std::vector<std::vector<float>> tile_luts(wtile_num * htile_num);

	// get transformation luts for each tile
	for (int i = 0; i < htile_num; i++)
	{
		for (int j = 0; j < wtile_num; j++)
		{
			cv::Rect roi(j * tile_size, i * tile_size, tile_size, tile_size);

			cv::Mat v_tile = channels[2](roi);
			tile_luts[i * wtile_num + j] = histogram(v_tile, m_clip_threshold);
		}
	}

	float* cdata = channels[2].ptr<float>();
	for (int i = tile_size / 2; i < input_final.rows - tile_size / 2; i++)
	{
		for (int j = tile_size / 2; j < input_final.cols - tile_size / 2; j++)
		{
			int v = static_cast<int>(cdata[i*input_final.cols+j] * 255.0f);

			int i_tile = (i - tile_size / 2) / tile_size;
			int j_tile = (j - tile_size / 2) / tile_size;

			int di_up = i - (i_tile * tile_size + tile_size / 2);
			int di_bottom = tile_size - di_up;
			int dj_left = j - (j_tile * tile_size + tile_size / 2);
			int dj_right = tile_size - dj_left;

			float c1 = tile_luts[i_tile * wtile_num + j_tile][v];
			float c2 = tile_luts[i_tile * wtile_num + j_tile + 1][v];
			float c3 = tile_luts[(i_tile + 1) * wtile_num + j_tile][v];
			float c4 = tile_luts[(i_tile + 1) * wtile_num + j_tile + 1][v];

			float v1 = lineer_interpolation(dj_left, dj_right, c1, c2);
			float v2 = lineer_interpolation(dj_left, dj_right, c3, c4);
			float vlast = lineer_interpolation(di_up, di_bottom, v1, v2);

			cdata[i*input_final.cols + j] = vlast;
		}
	}

	auto t2 = timer::now();
    std::cout << "my inner clahe time:" << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n";

	cv::merge(channels, input_final);

	// re-convert to bgr
	cv::Mat output_bgr, output_u8;
	cv::cvtColor(input_final, output_bgr, cv::COLOR_HSV2BGR);
	output_bgr.convertTo(output_u8, CV_8U, 255);

	cv::Rect original_roi(left, top, m_width, m_height);
	return output_u8(original_roi);
}
