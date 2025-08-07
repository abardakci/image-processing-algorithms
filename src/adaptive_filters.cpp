#include "adaptive_filters.hpp"

cv::Mat adaptive_mean_filter(cv::Mat &input, int filter_size, float noise_variance)
{
    CV_Assert(input.isContinuous() && !input.empty());
    CV_Assert(filter_size % 2 == 1 && filter_size > 0);
    CV_Assert(noise_variance >= 0.0f);

    int pad = filter_size / 2;
    cv::Mat padded(cv::Size(input.cols + 2 * pad, input.rows + 2 * pad), input.type());
    cv::copyMakeBorder(input, padded, pad, pad, pad, pad, cv::BORDER_REFLECT101);

    cv::Mat input_f;
    padded.convertTo(input_f, CV_32F, 1.0f / 255.0f);

    int h = input_f.rows;
    int w = input_f.cols;
    int filter_area = filter_size * filter_size;
    int k = filter_size / 2;

    cv::Mat output_f(input_f.size(), input_f.type());

    float *input_ptr = input_f.ptr<float>();
    float *output_ptr = output_f.ptr<float>();

    for (int i = pad; i < h - pad; ++i)
    {
        for (int j = pad; j < w - pad; ++j)
        {
            int offset = i * w + j;
            float current_pixel = input_ptr[offset];

            float patch_sum = 0.0f;
            for (int m = -k; m <= k; ++m)
            {
                for (int n = -k; n <= k; ++n)
                {
                    int patch_offset = (i + m) * w + j + n;
                    patch_sum += input_ptr[patch_offset];
                }
            }

            float patch_mean = patch_sum / filter_area;

            float patch_variance = 0.0f;
            for (int m = -k; m <= k; ++m)
            {
                for (int n = -k; n <= k; ++n)
                {
                    int patch_offset = (i + m) * w + j + n;
                    float diff = input_ptr[patch_offset] - patch_mean;
                    patch_variance += diff * diff;
                }
            }
            patch_variance /= (filter_area - 1);

            if (patch_variance > 1e-6f)
                output_ptr[offset] = current_pixel - (noise_variance / patch_variance) * (current_pixel - patch_mean);
            else
                output_ptr[offset] = patch_mean;
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
        if (pixel_value > min && pixel_value < max)
            return pixel_value;
        else
            return median;
    }
    return -1.0f;
}

static float find_adaptive_median(cv::Mat &input_f, int i, int j, int init_filter_size, int max_filter_size)
{
    int w = input_f.cols;
    int h = input_f.rows;
    float pixel_value = input_f.at<float>(i, j);

    int current_size = init_filter_size;
    while (current_size <= max_filter_size)
    {
        int k = current_size / 2;
        std::vector<float> patch;

        for (int m = -k; m <= k; ++m)
        {
            for (int n = -k; n <= k; ++n)
            {
                int patch_offset = (i + m) * w + j + n;
                patch.push_back(input_f.ptr<float>()[patch_offset]);
            }
        }

        std::sort(patch.begin(), patch.end());

        int n = patch.size();
        float median = patch[n / 2];
        float min = patch[0];
        float max = patch[n - 1];

        float val = levelA(pixel_value, min, median, max);
        if (val >= 0.0f)
            return val;

        current_size += 2;
    }

    return input_f.at<float>(i, j);
}

cv::Mat adaptive_median_filter(cv::Mat &input, int max_filter_size)
{
    CV_Assert(input.isContinuous() && !input.empty());
    CV_Assert(max_filter_size % 2 == 1 && max_filter_size > 1);

    int pad = max_filter_size / 2;
    cv::Mat padded(cv::Size(input.cols + 2 * pad, input.rows + 2 * pad), input.type());
    cv::copyMakeBorder(input, padded, pad, pad, pad, pad, cv::BORDER_REFLECT101);

    cv::Mat input_f;
    padded.convertTo(input_f, CV_32F, 1.0f / 255.0f);

    int h = input_f.rows;
    int w = input_f.cols;

    cv::Mat output_f(input_f.size(), input_f.type());
    float *output_ptr = output_f.ptr<float>();

    for (int i = pad; i < h - pad; ++i)
    {
        for (int j = pad; j < w - pad; ++j)
        {
            float val = find_adaptive_median(input_f, i, j, 3, max_filter_size);
            output_ptr[i * w + j] = val;
        }
    }

    cv::Mat output;
    output_f.convertTo(output, CV_8U, 255.0f);
    return output;
}
