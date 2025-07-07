#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <nlm.hpp>

using namespace std;

using timer = std::chrono::steady_clock;

const string assets_path = "/home/alper/code-repo/image_processing_cpp/assets/";
const string output_path = "/home/alper/code-repo/image_processing_cpp/outputs/";

const int max_width = 1280;
const int max_height = 720;

int main(int argc, char** argv)
{
    const int patch = stoi(argv[1]);
    const int window = stoi(argv[2]);
    const float t = stof(argv[3]);

    NLM nlm;
    cv::Mat input = cv::imread(assets_path + "noisyImage_Gaussian.jpg", cv::IMREAD_GRAYSCALE);
    input.convertTo(input, CV_32F);
    cv::Mat output = nlm.apply(input, patch, window, t);
    
    cv::Mat dst;
    cv::Mat src = cv::imread(assets_path + "noisyImage_Gaussian.jpg", cv::IMREAD_GRAYSCALE);
    fastNlMeansDenoising(src, dst, t, patch, window);
    cv::imwrite(output_path + "gaussian_opencv_nlm.jpg", dst);

    while (true)
    {
        cv::Mat display;
        cv::cvtColor(output, display, cv::COLOR_GRAY2BGR);
        display.convertTo(display, CV_8U);

        double scale_w = (double)max_width / display.cols;
        double scale_h = (double)max_height / display.rows;

        double scale = std::min(scale_w, scale_h);
        cv::imshow("CV", display);

        if (cv::waitKey(0) == 'q')
            break;
    }

    cv::destroyAllWindows();
    
    cv::imwrite(output_path + "noisyImage_Gaussian.jpg_nlm.jpg", output);

    return 0;
}
