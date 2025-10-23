#pragma once
// Minimal, stable ABI for Tensor/TensorBuffer/Allocator shared by processing and infer_engine.

#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstdlib> // <-- 新增，提供 std::malloc / std::free

namespace deploy {
namespace common {

enum class DType {
    FLOAT32,
    UINT8,
};

enum class Layout {
    NCHW,
    NHWC,
};

struct TensorDesc {
    std::vector<int> shape;
    DType dtype = DType::FLOAT32;
    Layout layout = Layout::NCHW;
};

struct TensorBuffer {
    // Ownership of data is via shared_ptr to allow zero-copy sharing.
    std::shared_ptr<void> data;
    size_t size_in_bytes = 0;
    TensorDesc desc;
};

class Allocator {
public:
    virtual ~Allocator() = default;
    // allocate/ free raw bytes. Return nullptr on failure.
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
};

class HostAllocator : public Allocator {
public:
    HostAllocator() = default;
    ~HostAllocator() override = default;
    void* allocate(size_t bytes) override {
        return std::malloc(bytes); // 保持 std:: 前缀
    }
    void deallocate(void* ptr) override {
        std::free(ptr);
    }
};

} // namespace common
} // namespace deploy