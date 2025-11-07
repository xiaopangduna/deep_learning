#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "type/Tensor.hpp"

namespace deploy::perception::processing {

using TensorPtr = std::shared_ptr<deploy::perception::types::Tensor>;

// Convert TensorPtr (host, HWC or {1,C,H,W}) -> cv::Mat (H,W,C).
// Returns true on success; err optional.
bool TensorToCvMat(const TensorPtr& t, cv::Mat& out, std::string* err = nullptr);

// Convert cv::Mat -> TensorPtr (host, HWC) with given dtype.
TensorPtr CvMatToTensor(const cv::Mat& m, deploy::perception::types::DType dtype);

} // namespace deploy::perception::processing