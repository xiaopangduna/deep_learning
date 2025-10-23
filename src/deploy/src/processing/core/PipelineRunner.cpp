#include "processing/Pipeline.hpp"

namespace deploy {
namespace processing {
namespace core {

// Minimal runner placeholder. Real runner may support async/streams.
bool run_pipeline(const Pipeline& pipeline,
                  const deploy::common::TensorBuffer& in,
                  deploy::common::TensorBuffer& out) {
    return pipeline.run(in, out);
}

} // namespace core
} // namespace processing
} // namespace deploy