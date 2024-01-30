import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 指定数据文件夹路径
data_folder = "data"

# 读取train.txt并划分训练集和验证集
with open("train.txt", "r") as file:
    lines = file.readlines()

# 跳过第一行
lines = lines[1:]

# 提取guid和tag信息
data = [line.strip().split(",") for line in lines]

# 根据guid生成文件路径
file_paths = [(os.path.join(data_folder, f"{guid}.txt"), os.path.join(data_folder, f"{guid}.jpg"), tag) for guid, tag in data]

# 划分训练集和验证集
train_data, val_data = train_test_split(file_paths, test_size=0.2, random_state=42)

# 加载文本数据
def load_text_data(file_path):
    with open(file_path, "r", encoding="ascii", errors="ignore") as file:
        text = file.read()
    return text

# 加载图像数据
def load_image_data(file_path):
    image = img_to_array(load_img(file_path, target_size=(224, 224))) / 255.0
    return image

# 使用LabelEncoder将文本标签转换为整数
label_encoder = LabelEncoder()
label_encoder.fit(["positive", "neutral", "negative"])

# 处理训练集数据
train_texts = [load_text_data(text_path) for text_path, _, _ in train_data]
train_images = [load_image_data(image_path) for _, image_path, _ in train_data]
train_labels = label_encoder.transform([tag for _, _, tag in train_data])

# 处理验证集数据
val_texts = [load_text_data(text_path) for text_path, _, _ in val_data]
val_images = [load_image_data(image_path) for _, image_path, _ in val_data]
val_labels = label_encoder.transform([tag for _, _, tag in val_data])

# 文本预处理
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
train_text_sequences = tokenizer.texts_to_sequences(train_texts)
val_text_sequences = tokenizer.texts_to_sequences(val_texts)
padded_train_texts = pad_sequences(train_text_sequences)
padded_val_texts = pad_sequences(val_text_sequences)

# 模型：只有文本输入
text_input = Input(shape=(None,))
embedding_layer = Embedding(input_dim=max_words, output_dim=16)(text_input)
lstm_layer = LSTM(32)(embedding_layer)
text_output = Dense(16, activation='relu')(lstm_layer)
output_text_only = Dense(3, activation='softmax')(text_output)

model_text_only = Model(inputs=text_input, outputs=output_text_only)
model_text_only.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_text_only.fit(x=padded_train_texts, y=train_labels, epochs=10, batch_size=32, validation_data=(padded_val_texts, val_labels))