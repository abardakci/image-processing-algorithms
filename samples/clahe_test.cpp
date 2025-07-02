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
    cv::setNumThreads(1);

    const int tile_size = stoi(argv[1]);
    const float clip_t = stof(argv[2]);
    string img_name = argv[3];

    cv::Mat input = cv::imread(assets_path + img_name);
    assert(!input.empty());
    
    cv::Mat output;

    CLAHE clahe;
    auto t1 = timer::now();
    output = clahe.apply(input, tile_size, clip_t);
    auto t2 = timer::now();
    cout << "my clahe time:" << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "\n";

    // CLAHE uygula
    cv::Mat img = cv::imread(assets_path + img_name, cv::IMREAD_GRAYSCALE);
    cv::Ptr<cv::CLAHE> cl = cv::createCLAHE();
    cl->setClipLimit(2.0);               // Kontrast limit değeri (default 40.0)
    cl->setTilesGridSize(cv::Size(8,8)); // Görüntüyü 8x8 bloklara bölerek işlem yapar
    cv::Mat dst;
    t1 = timer::now();
    cl->apply(img, dst);
    t2 = timer::now();
    cout << "opencv clahe time:" << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "\n";

    while (true)
    {
        cv::Mat display = output.clone();
                
        double scale_w = (double)max_width / display.cols;
        double scale_h = (double)max_height / display.rows;

        double scale = std::min(scale_w, scale_h);
        
        // cv::resize(display, display, cv::Size(display.cols * scale, display.rows * scale));
        
        cv::imshow("CV", display);

        if (cv::waitKey(0) == 'q')
            break;
    }

    cv::destroyAllWindows();

    cv::imwrite(output_path + "lena_clahe.jpg", output);

    return 0;
}
