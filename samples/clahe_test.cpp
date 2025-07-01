#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <clahe.hpp>

using namespace std;

using timer = std::chrono::steady_clock;

const string assets_path = "/home/alper/code-repo/image_processing_cpp/assets/";
const string output_path = "/home/alper/code-repo/image_processing_cpp/outputs/";

int main()
{
    cv::Mat input = cv::imread(assets_path + "lena_original.png");
    assert(!input.empty());
    
    cv::Mat output;

    CLAHE clahe;
    output = clahe.apply(input, 120, 0.2f);

    while (true)
    {
        cv::Mat display = output.clone();

        cv::imshow("CV", display);

        if (cv::waitKey(0) == 'q')
            break;
    }

    cv::destroyAllWindows();

    cv::imwrite(output_path + "lena_clahe.jpg", output);

    return 0;
}
