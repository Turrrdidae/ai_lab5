## 环境配置

使用Python3.9环境。需要安装以下依赖：

- numpy==1.26.3
- scikit_learn==1.3.0
- tensorflow==2.10.0

可以通过执行

```
pip install -r requirements.txt
```

以完成安装。

## 文件结构

```
|-- data	# 实验数据
|-- mfm.py	# 多模态融合模型代码
|-- only_pic.py		# 消融实验-只输入图片
|-- only_text.py	# 消融实验-只输入文本
|-- readme.md	
|-- requirements.txt	# 环境依赖
|-- test_without_label.txt	# 预测数据
|-- train.txt	# 训练数据
```

## 执行流程

下载代码，在`mfm.py`文件所在的文件夹下执行

```
python mfm.py
```

以进行多模态融合模型的训练与预测，预测结果输出为`predictions.txt`文件。

执行

```
python only_pic.py
```

以查看只输入图片的消融实验结果。

执行

```
python only_text.py
```

以查看只输入文本的消融实验结果。