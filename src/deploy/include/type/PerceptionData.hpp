#ifndef TYPE_PERCEPTION_DATA_HPP
#define TYPE_PERCEPTION_DATA_HPP

#include <vector>
#include <opencv2/opencv.hpp>

// 数据类型枚举（与各后端兼容）
enum class DataType {
    FP32,
    INT8,
    UINT8
};

// 张量结构（预处理→推理引擎的输入，推理引擎→后处理的输出）
struct Tensor {
    void* data = nullptr;  // 数据指针（需用户管理内存）
    std::vector<int> shape;  // 形状（NCHW格式）
    DataType dtype;  // 数据类型
    int bytes_per_elem() const {  // 每个元素的字节数
        switch (dtype) {
            case DataType::FP32: return 4;
            case DataType::INT8: case DataType::UINT8: return 1;
            default: return 0;
        }
    }
    size_t total_bytes() const {  // 总字节数
        size_t size = 1;
        for (int dim : shape) size *= dim;
        return size * bytes_per_elem();
    }
};

// 检测结果中的单个目标（含掩码）
struct Detection {
    cv::Rect2f bbox;  // 边界框（x,y,w,h，绝对坐标）
    float score = 0.0f;  // 置信度
    int class_id = -1;  // 类别ID
    cv::Mat mask;  // 掩码（二值图，与原图同尺寸或ROI尺寸）
};

// 分割模型的推理结果
struct SegmentationResult {
    std::vector<Detection> detections;  // 所有目标
    int img_h = 0;  // 原图高度
    int img_w = 0;  // 原图宽度
};

#endif  // TYPE_PERCEPTION_DATA_HPP