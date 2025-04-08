#ifndef YAML_NODE_WRAPPER_H
#define YAML_NODE_WRAPPER_H

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>

namespace LovelyUtils {
    /**
     * @brief 这是一个 YAML 节点包装类，用于简化 YAML 文件的读取、写入和操作。
     *
     * 使用示例:
     * @code
     * #include "yaml_node_wrapper.h"
     * #include <iostream>
     *
     * int main() {
     *     // 从文件加载 YAML 内容
     *     LovelyUtils::YamlNodeWrapper wrapper("example.yaml");
     *
     *     // 获取节点的值
     *     std::string value = wrapper.getValue<std::string>("key", "default");
     *     std::cout << "Value: " << value << std::endl;
     *
     *     // 设置节点的值
     *     wrapper.setValue("new_key", "new_value");
     *
     *     // 保存 YAML 内容到文件
     *     wrapper.saveToFile("output.yaml");
     *
     *     return 0;
     * }
     * @endcode
     */
    class YamlNodeWrapper {
    public:
        // 默认构造函数
        YamlNodeWrapper() : node_(YAML::Node()) {}

        // 从文件加载构造函数
        explicit YamlNodeWrapper(const std::string& filename) {
            loadFromFile(filename);
        }

        // 从字符串加载构造函数
        explicit YamlNodeWrapper(const char* yaml_content) {
            loadFromString(yaml_content);
        }

        // 获取内部的 YAML::Node
        YAML::Node& getNode() {
            return node_;
        }

        // 获取节点的值
        template <typename T>
        T getValue(const std::string& key, const T& default_value = T()) const {
            if (node_[key]) {
                try {
                    return node_[key].as<T>();
                } catch (const std::exception& e) {
                    std::cerr << "Error converting value for key '" << key << "': " << e.what() << std::endl;
                }
            }
            return default_value;
        }

        // 设置节点的值
        template <typename T>
        void setValue(const std::string& key, const T& value) {
            node_[key] = value;
        }

        // 保存 YAML 内容到文件
        bool saveToFile(const std::string& filename) const {
            try {
                std::ofstream fout(filename);
                if (!fout.is_open()) {
                    std::cerr << "Failed to open file for writing: " << filename << std::endl;
                    return false;
                }
                fout << node_;
                return true;
            } catch (const std::exception& e) {
                std::cerr << "Error saving to file '" << filename << "': " << e.what() << std::endl;
                return false;
            }
        }

        // 打印节点内容
        // void print(int indent = 0) const {
        //     // printNode(node_, indent);
        // }

        /**
         * @brief 将 YAML 节点内容以字符串形式返回
         * @return 包含 YAML 节点内容的字符串
         */
        std::string printToString() const {
            std::ostringstream oss;
            printNode(node_, 0, oss);
            return oss.str();
        }

    private:
        YAML::Node node_;

        // 从文件加载 YAML 内容
        void loadFromFile(const std::string& filename) {
            try {
                node_ = YAML::LoadFile(filename);
            } catch (const std::exception& e) {
                std::cerr << "Error loading file '" << filename << "': " << e.what() << std::endl;
            }
        }

        // 从字符串加载 YAML 内容
        void loadFromString(const char* yaml_content) {
            try {
                node_ = YAML::Load(yaml_content);
            } catch (const std::exception& e) {
                std::cerr << "Error loading YAML content from string: " << e.what() << std::endl;
            }
        }

        // // 递归打印节点内容
        // static void printNode(const YAML::Node& node, int indent) {
        //     if (node.IsScalar()) {
        //         std::cout << std::string(indent, ' ') << node.as<std::string>() << std::endl;
        //     } else if (node.IsSequence()) {
        //         for (const auto& item : node) {
        //             printNode(item, indent);
        //         }
        //     } else if (node.IsMap()) {
        //         for (const auto& pair : node) {
        //             std::cout << std::string(indent, ' ') << pair.first.as<std::string>() << ": ";
        //             printNode(pair.second, indent + 2);
        //         }
        //     }
        // }

        /**
         * @brief 递归打印节点内容到输出流
         * @param node 要打印的节点
         * @param indent 当前缩进级别
         * @param oss 输出流
         */
        static void printNode(const YAML::Node& node, int indent, std::ostringstream& oss) {
            if (node.IsScalar()) {
                oss << std::string(indent, ' ') << node.as<std::string>() << '\n';
            } else if (node.IsSequence()) {
                for (const auto& item : node) {
                    printNode(item, indent, oss);
                }
            } else if (node.IsMap()) {
                for (const auto& pair : node) {
                    oss << std::string(indent, ' ') << pair.first.as<std::string>() << ": ";
                    printNode(pair.second, indent + 2, oss);
                }
            }
        }
    };
} // namespace LovelyUtils

#endif // YAML_NODE_WRAPPER_H
