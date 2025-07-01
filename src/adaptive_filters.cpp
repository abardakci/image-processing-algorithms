#include "adaptive_filters.hpp"

static int get_mirror_offset(int i, int j, int img_w, int img_h, int filter_i, int filter_j)
{
    int step = filter_i * img_w + filter_j;
    bool b1 = i + filter_i >= img_h;
    bool b2 = i + filter_i < 0;
    bool b3 = j + filter_j >= img_w;
    bool b4 = j + filter_j < 0;

    if (b1 || b2)
        filter_i = -filter_i;

    if (b3 || b4)
        filter_j = -filter_j;

    int mirrored_offset = (i + filter_i) * img_w + j + filter_j;

    return mirrored_offset;
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

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            int offset = i * w + j;

            float patch_sum = 0.0f;
            float patch_mean = 0.0f;
            float patch_variance = 0.0f;

            float current_pixel = input_f.ptr<float>()[offset];

            for (int m = -k; m <= +k; ++m)
            {
                for (int n = -k; n <= +k; ++n)
                {
                    int patch_offset = get_mirror_offset(i, j, w, h, m, n);

                    patch_sum += input_f.ptr<float>()[patch_offset];
                }
            }

            patch_mean = patch_sum / filter_area;
            for (int m = -k; m <= +k; ++m)
            {
                for (int n = -k; n <= +k; ++n)
                {
                    int step = m * w + n;
                    int patch_offset = get_mirror_offset(i, j, w, h, m, n);

                    float distance = input_f.ptr<float>()[patch_offset] - patch_mean;
                    patch_variance += std::pow(distance, 2) / (filter_area - 1);
                }
            }

            output_f.ptr<float>()[offset] = current_pixel - (noise_variance / patch_variance) * (current_pixel - patch_mean);
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
    std::vector<float> patch(max_filter_size * max_filter_size);

    int offset = i * w + j;
    float pixel_value = input_f.ptr<float>()[offset];
    float median, min, max;

    while (filter_size < max_filter_size)
    {
        int filter_size = 3;
        int k = filter_size / 2;

        for (int m = -k; m <= k; ++m)
        {
            for (int n = -k; n <= k; ++n)
            {
                int patch_offset = get_mirror_offset(i, j, w, h, m, n);

                float val = input_f.ptr<float>()[patch_offset];
                patch.push_back(val);
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

// cv::Mat adaptive_median_filter(cv::Mat& input, int max_filter_size)
//{
//     CV_Assert(input.isContinuous() && !input.empty());
//
//     cv::Mat input_f;
//     input.convertTo(input_f, CV_32F, 1.0f/255.0f);
//
//     int filter_size = 3;
//
//     int h = input_f.rows;
//     int w = input_f.cols;
//
//     cv::Mat output_f;
//
//     int k = filter_size / 2;
//
//     std::vector<float> patch(max_filter_size * max_filter_size);
//     for (int i = 0; i < h; ++i)
//     {
//         for (int j = 0; j < w; ++j)
//         {
//             int offset = i * w + j;
//             output_f.ptr<float>()[offset] = find_adaptive_median(input_f, i, j, max_filter_size);
//         }
//     }
//
//     cv::Mat output;
//     output_f.convertTo(output, CV_8U, 255.0f);
//
//     return output;
// }
