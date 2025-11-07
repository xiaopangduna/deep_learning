#include "processing/Factory.hpp"
#include "processing/Op.hpp"
#include <vector>
#include <string>
#include <memory>

namespace deploy::perception::processing
{

    // minimal pipeline impl that applies ops in sequence to each tensor
    class PipelineImpl : public PreprocessPipeline
    {
    public:
        explicit PipelineImpl(std::vector<OpPtr> &&ops) : ops_(std::move(ops)) {}

        bool Run(const std::vector<TensorPtr> &in,
                 std::vector<TensorPtr> &out,
                 std::vector<std::string> *per_tensor_err = nullptr) const override
        {
            // out.clear();
            // out.resize(in.size());
            // if (per_tensor_err)
            //     per_tensor_err->clear();
            bool all_ok = true;

            // for (size_t i = 0; i < in.size(); ++i)
            // {
            //     TensorPtr cur = in[i];
            //     std::string local_err;
            //     bool ok = true;
            //     for (const auto &op : ops_)
            //     {
            //         TensorPtr next;
            //         // call Op::Run; pass nullptr for err if Op doesn't use it
            //         if (!op->Run(cur, next, nullptr))
            //         {
            //             ok = false;
            //             // if Op provides error reporting via other means, adapt here
            //             break;
            //         }
            //         cur = std::move(next);
            //     }
            //     if (!ok)
            //     {
            //         all_ok = false;
            //         out[i] = nullptr;
            //         if (per_tensor_err)
            //             per_tensor_err->push_back(local_err);
            //     }
            //     else
            //     {
            //         out[i] = std::move(cur);
            //         if (per_tensor_err)
            //             per_tensor_err->push_back(std::string());
            //     }
            // }
            return all_ok;
        }

    private:
        std::vector<OpPtr> ops_;
    };

    std::unique_ptr<PreprocessPipeline> BuildPreprocessPipeline(
        const std::vector<deploy::perception::types::Step> &steps,
        std::string *err)
    {

        std::vector<OpPtr> ops;
        ops.reserve(steps.size());

        for (size_t i = 0; i < steps.size(); ++i)
        {
            const auto &s = steps[i];
            std::string create_err;
            auto op = CreateOpByName(s.name, s.params, &create_err);
            if (!op)
            {
                if (err)
                {
                    *err = "BuildPreprocessPipeline: failed to create op '" + s.name +
                           "' at index " + std::to_string(i) + ": " + create_err;
                }
                return nullptr;
            }
            ops.push_back(std::move(op));
        }

        return std::make_unique<PipelineImpl>(std::move(ops));
    }

} // namespace deploy::perception::processing