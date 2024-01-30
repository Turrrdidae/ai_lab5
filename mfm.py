import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, LSTM, Embedding, Flatten
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

# 构建文本模型
text_input = Input(shape=(None,))
embedding_layer = Embedding(input_dim=max_words, output_dim=16)(text_input)
lstm_layer = LSTM(32)(embedding_layer)
text_output = Dense(16, activation='relu')(lstm_layer)

# 构建图像模型
image_input = Input(shape=(224, 224, 3))
flatten_layer = Flatten()(image_input)
image_output = Dense(16, activation='relu')(flatten_layer)

# 融合文本和图像模型
merged = concatenate([text_output, image_output])

# 构建输出层
output = Dense(3, activation='softmax')(merged)

# 构建模型
model = Model(inputs=[text_input, image_input], outputs=output)

# 编译模型
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x=[padded_train_texts, np.array(train_images)], y=train_labels, epochs=10, batch_size=32, validation_data=([padded_val_texts, np.array(val_images)], val_labels))

# 读取 test_without_label.txt
test_file_path = "test_without_label.txt"
with open(test_file_path, "r") as test_file:
    test_lines = test_file.readlines()

# 跳过第一行
test_lines = test_lines[1:]

# 提取 test_guids
test_guids = [line.strip().split(",")[0] for line in test_lines]

# 加载测试文本数据
test_texts = [load_text_data(os.path.join(data_folder, f"{guid}.txt")) for guid in test_guids]

# 文本预处理
test_text_sequences = tokenizer.texts_to_sequences(test_texts)
padded_test_texts = pad_sequences(test_text_sequences)

# 加载测试图像数据
test_images = [load_image_data(os.path.join(data_folder, f"{guid}.jpg")) for guid in test_guids]

# 预测标签
predictions = model.predict([padded_test_texts, np.array(test_images)])

# 将预测结果转换为标签
predicted_labels = np.argmax(predictions, axis=1)
predicted_tags = label_encoder.inverse_transform(predicted_labels)

# 将预测结果写入输出文件
output_file_path = "predictions.txt"
with open(output_file_path, "w") as output_file:
    output_file.write("guid,tag\n")
    for guid, tag in zip(test_guids, predicted_tags):
        output_file.write(f"{guid},{tag}\n")