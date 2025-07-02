#include "nlm.hpp"

// fix getting only square images
// add histogram clipping
// cover edges & corners

NLM::NLM() {}

NLM::~NLM() {}

cv::Mat NLM::apply(const cv::Mat &input, const int patch_size, const int window_size, const float h)
{
    CV_Assert(input.type() == CV_32F);

    m_width = input.cols;
    m_height = input.rows;

    const int f = patch_size / 2;
    const int t = window_size / 2;
    const int pad = t + f;

    cv::Mat padded(cv::Size(input.cols + 2*pad, input.rows + 2*pad), input.type());
    cv::copyMakeBorder(input, padded, pad, pad, pad, pad, cv::BORDER_REFLECT101);
    cv::Mat output = padded.clone();

    int padded_cols = padded.cols;
    float *data = padded.ptr<float>();
    for (int i = pad; i < pad + m_height; ++i)
    {
        for (int j = pad; j < pad + m_width; ++j)
        {   
            float denoised_pixel = 0.0f;
            int ij = i * padded_cols + j;
            int ji = j * padded_cols + i;

            float W = 0.0f;

            for (int m = -t; m < t; ++m)
            {
                for (int n = -t; n < t; ++n)
                {
                    float w = 0.0f;
                    for (int k = -f; k < f; ++k)
                    {
                        for (int l = -f; l < f; ++l)
                        {
                            int offset = (i + m + k) * padded_cols + (j + n + l);
                            w += std::exp(-std::pow(data[offset] - data[ij], 2) / (h*h));
                        }
                    }

                    W += w;
                    denoised_pixel += w * data[(i + m) * padded_cols + j + n];
                }
            }

            output.ptr<float>()[ij] = denoised_pixel / W;
        }
    }

    cv::Rect roi(pad, pad, m_width, m_height);
    return output(roi);
}