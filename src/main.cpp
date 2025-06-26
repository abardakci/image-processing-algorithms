#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "adaptive_filters.hpp"
#include "basic_filters.hpp"
#include "integral_img.hpp"
#include "slic.hpp"

using namespace std;

using timer = std::chrono::steady_clock;

const string assets_path = "C:/Code_Repo/ImageProcessing C++/assets/";
const string output_path = "C:/Code_Repo/ImageProcessing C++/outputs/";

const string image_path = assets_path + "noisyImage_Gaussian.jpg";

void testHistogramV1();
void testBoxFilterV1();
void testAdpMeanFilter();
void testSLIC();

int main()
{
    // testHistogramV1();
    // testBoxFilterV1();
    //testAdpMeanFilter();
    testSLIC();

    return 0;
}

void testSLIC()
{
    cout << "test SLIC" << endl;

    cv::Mat input = cv::imread(assets_path + "lena_original.png");

    const int num_sp = 120;
    const int threshold = 0.01f;
    cv::Mat output = SLIC(input, num_sp, threshold, 10);

    bool flag = true;
    while (true)
    {
        cv::Mat display = output.clone();

        cv::imshow("CV", display);
        int key = cv::waitKey(0);
        if (key == 'q')
            break;
    }

    cv::destroyAllWindows();

    cv::imwrite(output_path + "lena_slic.jpg", output);
}

void testIntegralImage()
{
    cout << "test convert_to_integral_image" << endl;

    cv::Mat input = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat inputc = input.clone();
    
    convert_to_integral_image(input);
    cv::Mat output;
    cv::integral(inputc, output, CV_8U);
}

void testAdpMeanFilter()
{
    cout << "hello" << endl;

    cv::Mat input = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    
    auto begin = timer::now();
    
    cv::Mat output = adaptive_mean_filter(input, 5, 0.001225f);

    auto end = timer::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << endl;
    
    bool flag = true;
    while (true)
    {
        cv::Mat display = output.clone();

        cv::imshow("CV", display);
        int key = cv::waitKey(0);
        if (key == 'q')
            break;
    }

    cv::destroyAllWindows();

    cv::imwrite("my_adaptive_mean_filter.jpg", output);
}

void testHistogramV1()
{
    cv::Mat input = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    
    auto begin = timer::now();
    
    cv::Mat output = histogramEqualization(input);

    auto end = timer::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << endl;

    cv::Mat output2;

    begin = timer::now();

    cv::equalizeHist(input, output2);
    
    end = timer::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << endl;
    
    bool flag = true;
    while (true)
    {
        cv::Mat display;
        if (flag)
        {
            display = output.clone();
            cv::putText(display, "Manual Histogram Equalization", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255), 2);
        }
        else
        {
            display = output2.clone();
            cv::putText(display, "OpenCV equalizeHist", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255), 2);
        }

        cv::imshow("CV", display);
        int key = cv::waitKey(0);
        if (key == 'q')
            flag = !flag;
        else
            break;
    }

    cv::destroyAllWindows();
    
    cv::Mat diff;
    cv::absdiff(output, output2, diff);
    double maxVal;
    cv::minMaxLoc(diff, nullptr, &maxVal);
    std::cout << maxVal << endl;
}

void testBoxFilterV1()
{
    cv::Mat input = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    
    auto begin = timer::now();
    
    cv::Mat output = boxFilter(input, 3);

    auto end = timer::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << endl;

    cv::Mat output2;

    begin = timer::now();

    cv::boxFilter(input, output2, -1, cv::Size(3,3));
    
    end = timer::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << endl;
    
    bool flag = true;
    while (true)
    {
        cv::Mat display;
        if (flag)
        {
            display = output.clone();
            cv::putText(display, "Manual", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255), 2);
        }
        else
        {
            display = output2.clone();
            cv::putText(display, "OpenCV", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255), 2);
        }

        cv::imshow("CV", display);
        int key = cv::waitKey(0);
        if (key == 'q')
            flag = !flag;
        else
            break;
    }

    cv::destroyAllWindows();
    
    cv::Mat diff;
    cv::absdiff(output, output2, diff);
    double maxVal;
    cv::minMaxLoc(diff, nullptr, &maxVal);
    std::cout << maxVal << endl;

}
