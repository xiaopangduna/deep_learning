./scripts/get_third_party_rknn.sh

# 合并模式：保留本地额外文件，覆盖远端同名文件
./scripts/get_third_party_rknn.sh -m

# 强制覆盖：删除并替换目标目录
./scripts/get_third_party_rknn.sh -f

# 指定目标目录和分支
./scripts/get_third_party_rknn.sh /opt/my_third_party main -m

# 指定 DEST 和 BRANCH 并强制覆盖
./scripts/get_third_party_rknn.sh third_party develop -f




cd /home/orangepi/HectorHuang/deep_learning/src/deploy

# 赋可执行权限（只需做一次）
chmod +x scripts/get_third_party_spdlog.sh

# 默认安装到 third_party/spdlog（若目标已存在会报错）
./scripts/get_third_party_spdlog.sh

# 合并模式：保留本地额外文件，覆盖相同文件
./scripts/get_third_party_spdlog.sh -m

# 强制替换：删除目标并用远端内容替换
./scripts/get_third_party_spdlog.sh -f

# 指定目标目录和分支/版本（示例：v1.11.0），并强制替换
./scripts/get_third_party_spdlog.sh third_party/spdlog v1.11.0 -f