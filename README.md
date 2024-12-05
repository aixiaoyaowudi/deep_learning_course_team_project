# 深度学习基础小组大作业仓库

本小组有以下成员: 肖子尧, 舒玉苗, 沈哲楷

## 文件架构

文件路径|描述
------|-----
`report/report.md`|项目报告 Markdown 源码
`report/images/`|项目报告中的图片文件
`scripts/process.py`|生成水印的预处理代码
`scripts/crawler.py`|从 Reddit 子论坛 EarthPorn 上爬取图像的代码
`scripts/convert.py`| labelme 格式标注文件到 Yolo 格式标注文件的转换脚本
`run_results/train_output.txt`|第三次训练时 nohup 记录的 `train.py` 的输出内容
`run_results/train_val/`|对训练数据运行 `detect.py` 的结果
`run_results/test_val/`|对测试数据运行 `val.py` 的结果
`model/watermark_detection.pt`|最终模型文件
`dataset/watermark/`|训练数据 & 测试数据
`dataset/watermark.yaml`|数据配置文件