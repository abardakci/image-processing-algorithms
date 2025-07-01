#include "slic.hpp"
#include <xmmintrin.h>

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

static void assignSamples(const cv::Mat &img, std::vector<Centroid> &centers, PixelCtx &pixel_ctx, int S)
{
	const int w = img.cols;
	const int h = img.rows;

	for (auto &centroid : centers)
	{
		for (int i = -S; i <= S; ++i)
		{
			for (int j = -S; j <= S; ++j)
			{
				int row_offset = centroid.xylab_.y() + i;
				int col_offset = centroid.xylab_.x() + j;

				if (row_offset >= h ||
					row_offset < 0 ||
					col_offset >= w ||
					col_offset < 0)
				{
					continue;
				}

				int pixel_offset = row_offset * w + col_offset;

				cv::Vec3f lab = img.at<cv::Vec3f>(row_offset, col_offset);
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

static void updateCentroids(const cv::Mat &img, std::vector<Centroid> &centers, PixelCtx &pixel_ctx)
{
	const int w = img.cols;
	const int h = img.rows;

	const int sp_num = centers.size();

	std::vector<int> C(sp_num, 0);
	std::vector<xylab> z(sp_num, 0.0f);

	for (auto &centroid : centers)
	{
		for (int i = 0; i < h; ++i)
		{
			for (int j = 0; j < w; ++j)
			{
				int offset = i * w + j;
				int idx = pixel_ctx.labels_[offset];

				C[idx] += 1;

				z[idx].setL(z[idx].L() + img.at<cv::Vec3f>(i, j)[0]);
				z[idx].setA(z[idx].A() + img.at<cv::Vec3f>(i, j)[1]);
				z[idx].setB(z[idx].B() + img.at<cv::Vec3f>(i, j)[2]);
				z[idx].setX(z[idx].x() + j);
				z[idx].setY(z[idx].y() + i);
			}
		}
	}

	for (int i = 0; i < z.size(); ++i)
	{
		z[i] /= C[i];
		centers[i].xylab_ = z[i];
	}
}

static cv::Mat postprocess(const cv::Mat &input, std::vector<Centroid> &centers, PixelCtx &pixel_ctx)
{
	const int w = input.cols;
	const int h = input.rows;

	assert(input.type() == CV_32FC3);
	cv::Mat output(cv::Size(w, h), input.type());

	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			int label = pixel_ctx.labels_[i * w + j];
			output.at<cv::Vec3f>(i, j)[0] = centers[label].xylab_.L();
			output.at<cv::Vec3f>(i, j)[1] = centers[label].xylab_.A();
			output.at<cv::Vec3f>(i, j)[2] = centers[label].xylab_.B();
		}
	}

	return output;
}

// input: input image to segment
// num_sp: number of super-pixels
// float T: threshold for convergence
// max_iter: maximum num of iterations
cv::Mat SLIC(const cv::Mat &input, int num_sp, float T, const int max_iter)
{
	// normalization & conversion to CIELAB
	cv::Mat bgr_float, lab_img, output;
	input.convertTo(bgr_float, CV_32F, 1.0f / 255.0f);
	cv::cvtColor(bgr_float, lab_img, cv::COLOR_BGR2Lab);

	// constants
	const int w = input.cols;
	const int h = input.rows;
	const int num_pixels = w * h;
	const int S = std::sqrt(num_pixels / num_sp);

	// objects required for slic
	std::vector<Centroid> centers;
	PixelCtx pixel_ctx(num_pixels);

	// initialize centroids
	for (int y = S / 2, id = 0; y < h; y += S)
	{
		for (int x = S / 2; x < w; x += S)
		{
			const cv::Vec3f &lab_pixel = lab_img.at<cv::Vec3f>(y, x);
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

	} while (iter++ < max_iter); // test for convergence

	// postprocess image
	output = postprocess(lab_img, centers, pixel_ctx);

	// re-convert to bgr
	cv::Mat output_bgr_float, output_bgr_u8;
	cv::cvtColor(output, output_bgr_float, cv::COLOR_Lab2BGR);
	output_bgr_float.convertTo(output_bgr_u8, CV_8U, 255);

	return output_bgr_u8;
}
