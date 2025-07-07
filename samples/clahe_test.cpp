#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <clahe.hpp>

using namespace std;

using timer = std::chrono::steady_clock;

const string assets_path = "/home/alper/code-repo/image_processing_cpp/assets/";
const string output_path = "/home/alper/code-repo/image_processing_cpp/outputs/";

const int max_width = 1280;
const int max_height = 720;

int main(int argc, char** argv)
{
    // default params
    int tile_size = 16;
    float clip_t = 0.01;
    string img_name = "badlena.png";

    if (argc == 4)
    {
        tile_size = stoi(argv[1]);
        clip_t = stof(argv[2]);
        img_name = argv[3];
    }

    cv::Mat input = cv::imread(assets_path + img_name, cv::IMREAD_GRAYSCALE);
    assert(!input.empty());
    
    auto t1 = timer::now();

    CLAHE clahe;
    clahe.apply(input, tile_size, clip_t);
    
    auto t2 = timer::now();
    cout << "my clahe time:" << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "\n";
    
    // // opencv clahe benchmark
    // cv::Mat img = cv::imread(assets_path + img_name, cv::IMREAD_GRAYSCALE);
    
    // cv::Ptr<cv::CLAHE> cl = cv::createCLAHE();
    // cl->setTilesGridSize(cv::Size(tile_size, tile_size)); // Görüntüyü 8x8 bloklara bölerek işlem yapar
    // cv::Mat dst;
    
    // t1 = timer::now();
    
    // cl->apply(img, dst);
    
    // t2 = timer::now();
    // cout << "opencv clahe time:" << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "\n";

    // // while (true)
    // // {
    // //     cv::Mat display = input.clone();
                
    // //     double scale_w = (double)max_width / display.cols;
    // //     double scale_h = (double)max_height / display.rows;

    // //     double scale = std::min(scale_w, scale_h);
        
    // //     // cv::resize(display, display, cv::Size(display.cols * scale, display.rows * scale));
        
    // //     cv::imshow("CV", display);

    // //     if (cv::waitKey(0) == 'q')
    // //         break;
    // // }

    // // cv::destroyAllWindows();

    cv::imwrite(output_path + "lena_clahe.jpg", input);

    return 0;
}
