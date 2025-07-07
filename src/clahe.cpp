#include "clahe.hpp"

CLAHE::CLAHE() {}

CLAHE::~CLAHE() {}

static void histogram(cv::Mat &channel, float clip_threshold, float *lut)
{
	CV_Assert(channel.type() == CV_8U);

	const int hist_size = 256;
	const int csize = channel.total();

	std::array<float, hist_size> pdf_arr{0};
	const uint8_t *cdata = channel.ptr<uint8_t>();

	cv::Mat pdf(1, hist_size, CV_32F, &pdf_arr);
	for (int i = 0; i < csize; ++i)
	{
		++pdf_arr[cdata[i]];
	}

	pdf /= csize;

	// clip
	float acc = 0.0f;
	for (int i = 0; i < hist_size; ++i)
	{
		float &h = pdf_arr[i];
		float excess = std::max(h - clip_threshold, 0.0f);
		acc += excess;
		h -= excess;
	}

	// redistribute
	float dist = acc / hist_size;
	pdf += dist;

	// CDF look-up table (lut)
	lut[0] = pdf_arr[0];
	for (int i = 1; i < hist_size; ++i)
	{
		lut[i] = lut[i - 1] + pdf_arr[i];
	}
}

static inline float lineer_interpolation(float dist_x1, float dist_x2, float x1_val, float x2_val)
{
	float sum = dist_x1 + dist_x2;
	return (dist_x1 / sum) * x2_val + (dist_x2 / sum) * x1_val;
}

cv::Mat CLAHE::apply(const cv::Mat &input, const int tile_size, float clip_threshold)
{
	CV_Assert(input.type() == CV_8U);

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
			histogram(v_tile, m_clip_threshold, &tile_luts[(i * wtile_num + j) * hist_size]);
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

			float c1 = tile_luts[(i_tile * wtile_num + j_tile) * hist_size + v];
			float c2 = tile_luts[(i_tile * wtile_num + j_tile + 1) * hist_size + v];
			float c3 = tile_luts[((i_tile + 1) * wtile_num + j_tile) * hist_size + v];
			float c4 = tile_luts[((i_tile + 1) * wtile_num + j_tile + 1) * hist_size + v];

			float v1 = lineer_interpolation(dj_left, dj_right, c1, c2);
			float v2 = lineer_interpolation(dj_left, dj_right, c3, c4);
			float vlast = lineer_interpolation(di_up, di_bottom, v1, v2);

			cdata[idx] = static_cast<uint8_t>(vlast * 255.0f);
		}
	}

	cv::Rect original_roi(left, top, m_width, m_height);
	return inputb(original_roi);
}
