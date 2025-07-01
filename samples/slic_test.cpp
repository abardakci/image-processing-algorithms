#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <slic.hpp>

using namespace std;

using timer = std::chrono::steady_clock;

const string assets_path = "/home/alper/code-repo/image_processing_cpp/assets/";
const string output_path = "/home/alper/code-repo/image_processing_cpp/outputs/";

int main()
{
    cv::Mat input = cv::imread(assets_path + "len_full.jpg");

    const int num_sp = 100;
    const int threshold = 0.01f;

    cv::Mat output = SLIC(input, num_sp, threshold, 10);

    while (true)
    {
        cv::Mat display = output.clone();
    
        cv::imshow("CV", display);
        
        if (cv::waitKey(0) == 'q')
            break;
    }

    cv::destroyAllWindows();
    
    cv::imwrite(output_path + "lena_full_slic.jpg", output);

    return 0;
}
