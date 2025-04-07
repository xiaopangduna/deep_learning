#ifndef YAML_CONFIG_H
#define YAML_CONFIG_H

#include "config.h"
#include <yaml-cpp/yaml.h>
#include <iostream>

class YamlConfig : public Config {
public:
    bool load(const std::string& file_path) override {
        try {
            config_yaml_ = YAML::LoadFile(file_path);
            return true;
        } catch (const YAML::Exception& e) {
            std::cerr << "Failed to load YAML file: " << e.what() << std::endl;
            return false;
        }
    }

    std::string get(const std::string& key, const std::string& default_value = "") const override {
        if (config_yaml_[key]) {
            return config_yaml_[key].as<std::string>();
        }
        return default_value;
    }

    std::shared_ptr<Config> get_nested(const std::string& key) const override {
        auto nested_config = std::make_shared<YamlConfig>();
        if (config_yaml_[key]) {
            nested_config->config_yaml_ = config_yaml_[key];
        }
        return nested_config;
    }

    bool contains(const std::string& key) const override {
        // 修改返回语句，使用 IsDefined() 方法
        return config_yaml_[key].IsDefined();
    }

    void print_all() const override {
        for (const auto& node : config_yaml_) {
            std::cout << node.first.as<std::string>() << ": " << node.second.as<std::string>() << std::endl;
        }
    }

private:
    YAML::Node config_yaml_;
};

#endif // YAML_CONFIG_H
