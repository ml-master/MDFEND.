#!/bin/bash

# 设置文件大小阈值（单位：字节）
THRESHOLD=$((100 * 1024 * 1024)) # 100 MB

# 查找大于阈值的文件并使用 Git LFS 跟踪
find . -type f -size +${THRESHOLD}c | while read -r file; do
    echo "Tracking large file: $file"
    git lfs track "$file"
done

# 添加 .gitattributes 文件并提交
git add .gitattributes
git commit -m "Track large files with Git LFS"
