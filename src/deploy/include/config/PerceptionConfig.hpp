// #ifndef PERCEPTION_CONFIG_HPP
// #define PERCEPTION_CONFIG_HPP
#pragma once
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <string>
#include <sstream>


template<typename T>
void loadValueFromYaml(const YAML::Node &node, const std::string& key, T& value) {
    if (node[key]) {
        try {
            value = node[key].as<T>();
        } catch (const YAML::Exception& e) {
            std::cerr << "Error loading value for key " << key << ": " << e.what() << std::endl;
        }
    }
}
// 基类：表示一个通用的配置
struct PerceptionConfig
{
    virtual ~PerceptionConfig() =default;
    // 从 YAML::Node 加载配置，纯虚函数，由派生类实现
    virtual void loadFromYamlNode(const YAML::Node &node) = 0;
    virtual void printConfig() const = 0;
    virtual std::string configToString() const = 0;
};


// #endif // PERCEPTION_CONFIG_HPP