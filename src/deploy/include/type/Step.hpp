#pragma once
#include <map>
#include <string>

namespace deploy::perception::types
{

    // 参数映射类型
    using Params = std::map<std::string, std::string>;

    // 描述用于构建 pipeline 的单个步骤（算子名 + 参数）
    struct Step
    {
        std::string name;
        Params params;
    };
}