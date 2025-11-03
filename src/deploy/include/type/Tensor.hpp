#pragma once
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace deploy::perception::types {

enum class DType { UINT8, FLOAT32, UNKNOWN };

inline std::size_t BytesPerElement(DType dt) {
    switch (dt) {
    case DType::UINT8:  return 1;
    case DType::FLOAT32:return 4;
    default:            return 0;
    }
}

// 轻量 Tensor/Buffer 抽象：data 为不透明指针，device 标识内存位置
struct Tensor {
    std::string device = "cpu";            // "cpu", "cuda:0", "rknn:0", ...
    DType dtype = DType::UNKNOWN;
    std::vector<int> shape;                // e.g. {N,C,H,W} 或自定义约定
    std::size_t byte_size = 0;
    void* data = nullptr;                  // host pointer 或 device-specific handle (opaque)
    std::function<void(void*)> deleter;    // optional custom deleter

    // Allocate host memory and return shared_ptr managing it.
    // Throws std::invalid_argument on invalid dtype/shape or std::bad_alloc on OOM.
    static std::shared_ptr<Tensor> AllocateHost(DType dt, const std::vector<int>& shape) {
        if (dt == DType::UNKNOWN) throw std::invalid_argument("AllocateHost: unknown dtype");
        if (shape.empty()) throw std::invalid_argument("AllocateHost: empty shape");
        std::size_t nelems = 1;
        for (int d : shape) {
            if (d < 0) throw std::invalid_argument("AllocateHost: negative dim");
            nelems = nelems * static_cast<std::size_t>(d);
        }
        std::size_t bpe = BytesPerElement(dt);
        if (bpe == 0) throw std::invalid_argument("AllocateHost: unsupported dtype");
        std::size_t bytes = nelems * bpe;
        void* ptr = std::malloc(bytes);
        if (!ptr) throw std::bad_alloc();
        auto t = std::make_shared<Tensor>();
        t->device = "cpu";
        t->dtype = dt;
        t->shape = shape;
        t->byte_size = bytes;
        t->data = ptr;
        t->deleter = [](void* p){ std::free(p); };
        return t;
    }

    // Wrap an external buffer (host or device) without copying; caller provides deleter (may be nullptr).
    static std::shared_ptr<Tensor> WrapExternal(void* external_ptr,
                                                DType dt,
                                                const std::vector<int>& shape,
                                                const std::string& device = "cpu",
                                                std::function<void(void*)> deleter = nullptr) {
        if (!external_ptr) throw std::invalid_argument("WrapExternal: null pointer");
        std::size_t nelems = 1;
        for (int d : shape) {
            if (d < 0) throw std::invalid_argument("WrapExternal: negative dim");
            nelems = nelems * static_cast<std::size_t>(d);
        }
        std::size_t bpe = BytesPerElement(dt);
        if (bpe == 0) throw std::invalid_argument("WrapExternal: unsupported dtype");
        auto t = std::make_shared<Tensor>();
        t->device = device;
        t->dtype = dt;
        t->shape = shape;
        t->byte_size = nelems * bpe;
        t->data = external_ptr;
        t->deleter = std::move(deleter);
        return t;
    }

    // Release owned buffer immediately (idempotent)
    void Release() {
        if (data && deleter) {
            try { deleter(data); } catch (...) {}
        }
        data = nullptr;
        deleter = nullptr;
        byte_size = 0;
    }

    ~Tensor() {
        Release();
    }
};

using TensorPtr = std::shared_ptr<Tensor>;

} // namespace deploy::perception::types