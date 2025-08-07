#include "slic.hpp"

static constexpr float INF = std::numeric_limits<float>::max();

PixelCtx::PixelCtx(int size) : labels_(size, -1), distances_(size, INF) {}

static float distance(xylab &p1, xylab &p2, float dcm, float dsm)
{
	cv::Vec3f v1(p1.L() - p2.L(), p1.A() - p2.A(), p1.B() - p2.B());
	float dc = cv::norm(v1);

	float dx = p1.x() - p2.x();
	float dy = p1.y() - p2.y();
	float ds = std::sqrt(dx * dx + dy * dy);

	float D = std::sqrt((dc * dc) / (dcm * dcm) + (ds * ds) / (dsm * dsm));

	return D;
}

SLIC::SLIC() {}

void SLIC::assignSamples(cv::Mat &img, std::vector<Centroid> &centers, PixelCtx &pixel_ctx, int S)
{
	cv::Vec3f *ptr = img.ptr<cv::Vec3f>();
	for (auto &centroid : centers)
	{
		for (int i = -S; i <= S; ++i)
		{
			for (int j = -S; j <= S; ++j)
			{
				int row_offset = centroid.xylab_.y() + i;
				int col_offset = centroid.xylab_.x() + j;

				if (row_offset >= m_height ||
					row_offset < 0 ||
					col_offset >= m_width ||
					col_offset < 0)
				{
					continue;
				}

				int pixel_offset = row_offset * m_width + col_offset;

				cv::Vec3f lab = ptr[row_offset * m_width + col_offset];
				xylab current_pixel(col_offset, row_offset, lab[0], lab[1], lab[2]);

				float current_distance = distance(centroid.xylab_, current_pixel, 20.0f, S);

				if (current_distance < pixel_ctx.distances_[pixel_offset])
				{
					pixel_ctx.labels_[pixel_offset] = centroid.label_;
					pixel_ctx.distances_[pixel_offset] = current_distance;
				}
			}
		}
	}
}

void SLIC::updateCentroids(cv::Mat &img, std::vector<Centroid> &centers, PixelCtx &pixel_ctx)
{
	const int sp_num = centers.size();

	std::vector<int> C(sp_num, 0);
	std::vector<xylab> z(sp_num, 0.0f);

	cv::Vec3f *ptr = img.ptr<cv::Vec3f>();

	for (int i = 0; i < m_height; ++i)
	{
		for (int j = 0; j < m_width; ++j)
		{
			int offset = i * m_width + j;
			int idx = pixel_ctx.labels_[offset];

			C[idx] += 1;

			z[idx].setL(z[idx].L() + ptr[i * m_width + j][0]);
			z[idx].setA(z[idx].A() + ptr[i * m_width + j][1]);
			z[idx].setB(z[idx].B() + ptr[i * m_width + j][2]);
			z[idx].setX(z[idx].x() + j);
			z[idx].setY(z[idx].y() + i);
		}
	}

	int zsize = z.size(); 
	for (int i = 0; i < zsize; ++i)
	{
		z[i] /= C[i];
		centers[i].xylab_ = z[i];
	}
}

cv::Mat SLIC::postprocess(const cv::Mat &input, std::vector<Centroid> &centers, PixelCtx &pixel_ctx)
{
	CV_Assert(input.type() == CV_32FC3);
	cv::Mat output(cv::Size(m_width, m_height), input.type());

	cv::Vec3f *out = output.ptr<cv::Vec3f>();
	for (int i = 0; i < m_height; ++i)
	{
		for (int j = 0; j < m_width; ++j)
		{
			int label = pixel_ctx.labels_[i * m_width + j];
			out[i * m_width + j][0] = centers[label].xylab_.L();
			out[i * m_width + j][1] = centers[label].xylab_.A();
			out[i * m_width + j][2] = centers[label].xylab_.B();
		}
	}

	return output;
}

// input: input image to segment
// num_sp: number of super-pixels
// float T: threshold for convergence
// max_iter: maximum num of iterations
cv::Mat SLIC::apply(cv::Mat &input, int num_sp, float T, const int max_iter)
{
	// normalization & conversion to CIELAB
	cv::Mat bgr_float, lab_img, output;
	input.convertTo(bgr_float, CV_32F, 1.0f / 255.0f);
	cv::cvtColor(bgr_float, lab_img, cv::COLOR_BGR2Lab);

	// constants
	m_width = input.cols;
	m_height = input.rows;
	const int num_pixels = m_width * m_height;
	const int S = std::sqrt(num_pixels / num_sp);

	// objects required for slic
	std::vector<Centroid> centers;
	PixelCtx pixel_ctx(num_pixels);

	// initialize centroids
	cv::Vec3f *lab = lab_img.ptr<cv::Vec3f>();
	for (int y = S / 2, id = 0; y < m_height; y += S)
	{
		for (int x = S / 2; x < m_width; x += S)
		{
			const cv::Vec3f &lab_pixel = lab[y * m_width + x];
			centers.emplace_back(xylab(x, y, lab_pixel[0], lab_pixel[1], lab_pixel[2]), id++);
		}
	}

	int iter = 0;
	do
	{
		// assign samples to cluster centers
		assignSamples(lab_img, centers, pixel_ctx, S);

		// update centroids
		updateCentroids(lab_img, centers, pixel_ctx);

	} while (iter++ < max_iter); // test for convergence (not yet implemented)

	// postprocess image
	output = postprocess(lab_img, centers, pixel_ctx);

	// re-convert to bgr
	cv::Mat output_bgr_float, output_bgr_u8;
	cv::cvtColor(output, output_bgr_float, cv::COLOR_Lab2BGR);
	output_bgr_float.convertTo(output_bgr_u8, CV_8U, 255);

	return output_bgr_u8;
}
