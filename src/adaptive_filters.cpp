#include "adaptive_filters.hpp"

static int get_mirror_offset(int i, int j, int img_w, int img_h, int filter_i, int filter_j)
{
    int ni = i + filter_i;
    int nj = j + filter_j;

    if (ni < 0)
        ni = -ni;
    else if (ni >= img_h)
        ni = 2 * img_h - ni - 1;

    if (nj < 0)
        nj = -nj;
    else if (nj >= img_w)
        nj = 2 * img_w - nj - 1;

    return ni * img_w + nj;
}


cv::Mat adaptive_mean_filter(cv::Mat &input, int filter_size, float noise_variance)
{
    CV_Assert(input.isContinuous() && !input.empty());
    CV_Assert(filter_size % 2 == 1 && filter_size > 0);
    CV_Assert(noise_variance >= 0.0f);

    cv::Mat input_f;
    input.convertTo(input_f, CV_32F, 1.0f / 255.0f);

    int filter_area = filter_size * filter_size;

    int h = input_f.rows;
    int w = input_f.cols;

    cv::Mat output_f(input_f.size(), input_f.type());

    int k = filter_size / 2;

    float *input_ptr = input_f.ptr<float>();
    float *output_ptr = output_f.ptr<float>();
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            int offset = i * w + j;

            float patch_sum = 0.0f;
            float patch_mean = 0.0f;
            float patch_variance = 0.0f;

            float current_pixel = input_ptr[offset];

            for (int m = -k; m <= +k; ++m)
            {
                for (int n = -k; n <= +k; ++n)
                {
                    int patch_offset = get_mirror_offset(i, j, w, h, m, n);

                    patch_sum += input_ptr[patch_offset];
                }
            }

            patch_mean = patch_sum / filter_area;
            for (int m = -k; m <= +k; ++m)
            {
                for (int n = -k; n <= +k; ++n)
                {
                    int step = m * w + n;
                    int patch_offset = get_mirror_offset(i, j, w, h, m, n);

                    float distance = input_ptr[patch_offset] - patch_mean;
                    patch_variance += std::pow(distance, 2) / (filter_area - 1);
                }
            }

            output_ptr[offset] = current_pixel - (noise_variance / patch_variance) * (current_pixel - patch_mean);
        }
    }

    cv::Mat output;
    output_f.convertTo(output, CV_8U, 255.0f);

    return output;
}

static float levelA(float pixel_value, float min, float median, float max)
{
    if (min < median && median < max)
    {
        if (min < pixel_value && pixel_value < max)
        {
            return pixel_value;
        }
        else
        {
            return median;
        }
    }

    return -1.0f;
}

static float find_adaptive_median(cv::Mat &input_f, int i, int j, int filter_size, int max_filter_size)
{
    int w = input_f.cols;
    int h = input_f.rows;

    float pixel_value = input_f.at<float>(i, j);
    float median = 0, min = 0, max = 0;

    while (filter_size <= max_filter_size)
    {
        int k = filter_size / 2;
        std::vector<float> patch;

        for (int m = -k; m <= k; ++m)
        {
            for (int n = -k; n <= k; ++n)
            {
                int patch_offset = get_mirror_offset(i, j, w, h, m, n);
                patch.push_back(input_f.ptr<float>()[patch_offset]);
            }
        }

        std::sort(patch.begin(), patch.end());

        int n = patch.size();
        median = patch[n / 2];
        min = patch[0];
        max = patch[n - 1];

        float val = levelA(pixel_value, min, median, max);
        if (val > 0.0f)
        {
            return val;
        }

        filter_size += 2;
    }

    return median;
}

cv::Mat adaptive_median_filter(cv::Mat &input, int max_filter_size)
{
    CV_Assert(input.isContinuous() && !input.empty());
    CV_Assert(max_filter_size % 2 == 1 && max_filter_size > 1);

    cv::Mat input_f;
    input.convertTo(input_f, CV_32F, 1.0f / 255.0f);

    int h = input_f.rows;
    int w = input_f.cols;

    cv::Mat output_f(input_f.size(), input_f.type());

    float *output_ptr = output_f.ptr<float>();
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            float val = find_adaptive_median(input_f, i, j, 3, max_filter_size);
            if (val < 0.0f)
            {
                val = input_f.at<float>(i, j);
            }

            output_ptr[i * w + j] = val;
        }
    }

    cv::Mat output;
    output_f.convertTo(output, CV_8U, 255.0f);
    return output;
}
