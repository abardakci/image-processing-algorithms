#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <adaptive_filters.hpp>

using namespace std;

using timer = std::chrono::steady_clock;

const string assets_path = "/home/alper/code-repo/image_processing_cpp/assets/";
const string output_path = "/home/alper/code-repo/image_processing_cpp/outputs/";

int main(int argc, char** argv)
{
    string img_name = argv[1];
    float max_win = stoi(argv[2]);

    cv::Mat input = cv::imread(assets_path + img_name, cv::IMREAD_GRAYSCALE);
    cv::Mat output = adaptive_median_filter(input, max_win);

    while (true)
    {
        cv::Mat display = output.clone();
        cv::imshow("CV", display);

        if (cv::waitKey(0) == 'q')
            break;
    }

    cv::destroyAllWindows();
    
    cv::imwrite(output_path + "lena_SaltPepper_median.jpg", output);

    return 0;
}
