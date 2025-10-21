// 正确的main函数定义（必须是全局作用域，不能在命名空间内）
#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

int main()
{ // 函数名必须是main，返回类型必须是int
    std::cout << "Yolov5 seg demo is running..." << std::endl;

    // 1. 打印OpenCV版本（验证头文件是否正常包含）
    std::cout << "OpenCV版本: " << CV_VERSION << std::endl;

    // 2. 读取测试图像（验证图像读取功能）
    cv::Mat img = cv::imread("/home/xiaopangdun/project/deep_learning/src/deploy/examples/yolov5_seg_example/test_image.jpg"); // 假设该文件存在于examples/yolov5_seg_example目录
    if (img.empty())
    {
        std::cerr << "错误：无法读取图像 test_image.jpg" << std::endl;
        return -1;
    }
    std::cout << "图像读取成功，尺寸: " << img.rows << "x" << img.cols << std::endl;

    // 3. 简单图像处理（验证OpenCV功能是否正常）
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // 转为灰度图
    std::cout << "图像处理成功（转为灰度图）" << std::endl;

    // 5. （可选，若有显示器）显示图像
    cv::namedWindow("Test Image", cv::WINDOW_NORMAL);
    cv::imshow("Test Image", img);
    cv::waitKey(0); // 按任意键关闭窗口
    cv::destroyAllWindows();

    std::cout << "OpenCV功能验证成功！" << std::endl;

    YAML::Node node = YAML::Load("{name: ChatGPT, type: AI}");
    std::cout << "name: " << node["name"].as<std::string>() << std::endl;
    std::cout << "type: " << node["type"].as<std::string>() << std::endl;
    return 0;
    return 0;
}