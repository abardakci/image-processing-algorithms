#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <slic.hpp>

using namespace std;

using timer = std::chrono::steady_clock;

const string assets_path = "/home/alper/code-repo/image_processing_cpp/assets/";
const string output_path = "/home/alper/code-repo/image_processing_cpp/outputs/";

const int max_width = 1280;
const int max_height = 720;

int main(int argc, char** argv)
{
    const int num_sp = stoi(argv[1]);
    const float threshold = stof(argv[2]);
    string img_name = argv[3];

    cv::Mat input = cv::imread(assets_path + img_name);
    cv::Mat output = SLIC(input, num_sp, threshold, 10);

    while (true)
    {
        cv::Mat display = output.clone();
                
        double scale_w = (double)max_width / display.cols;
        double scale_h = (double)max_height / display.rows;

        double scale = std::min(scale_w, scale_h);
        cv::imshow("CV", display);

        if (cv::waitKey(0) == 'q')
            break;
    }

    cv::destroyAllWindows();
    
    cv::imwrite(output_path + "lena_full_slic.jpg", output);

    return 0;
}
