#pragma once

#include "processing/Transform.hpp"
#include <vector>
#include <memory>

namespace deploy {
namespace processing {

class Pipeline {
public:
    Pipeline() = default;
    ~Pipeline() = default;

    void add_transform(TransformPtr t) { transforms_.push_back(std::move(t)); }

    // Run pipeline synchronously for single sample.
    bool run(const Buffer& in, Buffer& out) const {
        Buffer cur = in;
        Buffer tmp;
        for (const auto& t : transforms_) {
            tmp = {}; // reset
            if (!t->apply(cur, tmp)) return false;
            cur = std::move(tmp);
        }
        out = std::move(cur);
        return true;
    }

    size_t size() const { return transforms_.size(); }

private:
    std::vector<TransformPtr> transforms_;
};

} // namespace processing
} // namespace deploy