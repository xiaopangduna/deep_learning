#include "processing/Op.hpp"
#include <mutex>
#include <unordered_map>
#include <string>

namespace deploy::perception::processing {

namespace {
std::unordered_map<std::string, OpCreateFn>& Registry() {
    static std::unordered_map<std::string, OpCreateFn> g;
    return g;
}
std::mutex& RegistryMutex() {
    static std::mutex m;
    return m;
}
} // namespace

bool RegisterOpFactory(const std::string& op_name, OpCreateFn fn) {
    if (!fn) return false;
    std::lock_guard<std::mutex> lk(RegistryMutex());
    Registry()[op_name] = std::move(fn);
    return true;
}

OpPtr CreateOpByName(const std::string& op_name, const Params& params, std::string* err) {
    std::lock_guard<std::mutex> lk(RegistryMutex());
    auto it = Registry().find(op_name);
    if (it == Registry().end()) {
        if (err) *err = "no factory registered for op: " + op_name;
        return nullptr;
    }
    try {
        return it->second(params, err);
    } catch (const std::exception& e) {
        if (err) *err = std::string("exception in factory for '") + op_name + "': " + e.what();
        return nullptr;
    } catch (...) {
        if (err) *err = std::string("unknown exception in factory for '") + op_name + "'";
        return nullptr;
    }
}

} // namespace deploy::perception::processing