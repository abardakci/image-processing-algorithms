#include "clahe.hpp"

CLAHE::CLAHE() {}

CLAHE::~CLAHE() {}

static std::vector<int> calcHist(cv::Mat &channel, int bin_size)
{
	CV_Assert(channel.type() == CV_8U);

	std::vector<int> hist(bin_size, 0);

	int total = channel.rows * channel.cols;

	const uint8_t *data = channel.data; // channel.data zaten uint8_t*
	for (int i = 0; i < total; ++i)
	{
		++hist[data[i]];
	}

	return hist;
}

static std::vector<float> histogram(cv::Mat &channel, float clip_threshold)
{
	CV_Assert(channel.type() == CV_8U && channel.isContinuous());

	const int hist_size = 256;

	std::vector<int> hist_int = calcHist(channel, 256);

	std::vector<float> pdf(hist_size);

	// normalize to PDF
	int total = channel.total();
	for (int i = 0; i < hist_size; ++i)
	{
		pdf[i] = (float)hist_int[i] / total;
	}

	// clip
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

	// redistribute
	float dist = acc / pdf.size();
	for (int i = 0; i < pdf.size(); ++i)
	{
		pdf[i] += dist;
	}

	// CDF look-up table (lut)
	std::vector<float> lut(hist_size, 0.0f);
	lut[0] = pdf[0];
	for (int i = 1; i < hist_size; ++i)
	{
		lut[i] += lut[i - 1] + pdf[i];
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

	int wpad = tile_size - (m_width % tile_size);
	int left = wpad / 2;
	int right = wpad - left;

	int hpad = tile_size - (m_height % tile_size);
	int top = hpad / 2;
	int down = hpad - top;

	cv::Mat inputb(cv::Size(m_width + wpad, m_height + hpad), input.type());
	cv::copyMakeBorder(input, inputb, top, down, left, right, CV_HAL_BORDER_REFLECT_101);

	const int wtile_num = inputb.cols / tile_size;
	const int htile_num = inputb.rows / tile_size;

	const int hist_size = 256;
	std::vector<float> tile_luts(wtile_num * htile_num * hist_size);

	// get transformation luts for each tile
	for (int i = 0; i < htile_num; i++)
	{
		for (int j = 0; j < wtile_num; j++)
		{
			cv::Rect roi(j * tile_size, i * tile_size, tile_size, tile_size);

			cv::Mat v_tile = inputb(roi);
			tile_luts[i * wtile_num * hist_size + j * hist_] = histogram(v_tile, m_clip_threshold);
		}
	}

	uint8_t *cdata = inputb.data;
	const int inputw = inputb.cols;
	for (int i = tile_size / 2; i < inputb.rows - tile_size / 2; i++)
	{
		for (int j = tile_size / 2; j < inputw - tile_size / 2; j++)
		{
			const int idx = i * inputw + j;
			int v = static_cast<int>(cdata[idx]);

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

			cdata[idx] = static_cast<uint8_t>(vlast * 255.0f);
		}
	}

	cv::Rect original_roi(left, top, m_width, m_height);
	return inputb(original_roi);
}
