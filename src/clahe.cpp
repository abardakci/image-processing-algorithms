#include "clahe.hpp"

CLAHE::CLAHE() {}

CLAHE::~CLAHE() {}

static void histogram(const cv::Mat &channel, float clip_threshold, float *lut)
{
	CV_Assert(channel.type() == CV_8U);
	
	const int hist_size = 256;
	const float range[] = {0.f, 256.f};
	const float* hist_range = range;

	// OpenCV histogram (CV_32F)
	cv::Mat hist;
	cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range, true, false);
	float *h = hist.ptr<float>(); 

	// normalize to PDF (divide by total pixel count)
	hist /= static_cast<float>(channel.total());

	// clip
	float acc = 0.0f;
	for (int i = 0; i < hist_size; ++i)
	{
		float excess = std::max(h[i] - clip_threshold, 0.0f);
		acc += excess;
		h[i] -= excess;
	}

	// redistribute
	float dist = acc / hist_size;
	for (int i = 0; i < hist_size; ++i)
		h[i] += dist;

	// CDF look-up table (lut)
	lut[0] = h[0];
	for (int i = 1; i < hist_size; ++i)
	{
		lut[i] = lut[i - 1] + h[i];
	}
}

static inline float lineer_interpolation(float dist_x1, float dist_x2, float x1_val, float x2_val)
{
	float sum = dist_x1 + dist_x2;
	return (dist_x1 / sum) * x2_val + (dist_x2 / sum) * x1_val;
}

void CLAHE::apply(const cv::Mat &input, const int tile_size, float clip_threshold)
{
	CV_Assert(input.type() == CV_8U && input.isContinuous() && !input.empty());

	m_height = input.rows;
	m_width = input.cols;
	m_clip_threshold = clip_threshold;

	const int wtile_num = m_width / tile_size;
	const int htile_num = m_height / tile_size;
	const int hist_size = 256;

	auto t1 = timer::now();
	std::vector<float> tile_luts(wtile_num * htile_num * hist_size);

	// get transformation luts for each tile

	#pragma omp parallel for
	for (int index = 0; index < htile_num * wtile_num; ++index)
	{
		int i = index / wtile_num;
		int j = index % wtile_num;

		cv::Rect roi(j * tile_size, i * tile_size, tile_size, tile_size);
		cv::Mat v_tile = input(roi);

		histogram(v_tile, m_clip_threshold, &tile_luts[index * hist_size]);
	}	

    auto t2 = timer::now();
    std::cout << "histogram loop: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\n";

	uint8_t *cdata = input.data;
	
	t1 = timer::now();

	const int tdiv2 = tile_size / 2;
	const int edge_i = m_width - tdiv2;
	const int edge_j = m_width - tdiv2;
	
	#pragma omp parallel for
	for (int i = tdiv2; i < edge_i; i++)
	{
		for (int j = tdiv2; j < edge_j; j++)
		{
			const int idx = i * m_width + j;
			int v = static_cast<int>(cdata[idx]);

			int i_tile = (i - tdiv2) / tile_size;
			int j_tile = (j - tdiv2) / tile_size;

			int di_up = i - (i_tile * tile_size + tdiv2);
			int di_bottom = tile_size - di_up;
			int dj_left = j - (j_tile * tile_size + tdiv2);
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

	t2 = timer::now();
    std::cout << "interpolation loop: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\n";
}
