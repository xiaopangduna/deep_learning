#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <memory>

class Config {
public:
    virtual ~Config() = default;

    // 加载配置文件
    virtual bool load(const std::string& file_path) = 0;

    // 获取指定键的值
    virtual std::string get(const std::string& key, const std::string& default_value = "") const = 0;

    // 获取嵌套的配置
    virtual std::shared_ptr<Config> get_nested(const std::string& key) const = 0;

    // 检查配置项是否存在
    virtual bool contains(const std::string& key) const = 0;

    // 打印所有配置项
    virtual void print_all() const = 0;
};

#endif // CONFIG_H
