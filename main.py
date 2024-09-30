import tensorflow as tf 
from config import * 

# モデルの構成: Transformerベース (例)
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=100000, output_dim=256),
    tf.keras.layers.TransformerEncoderLayer(num_heads=8, units=5120), 
    tf.keras.layers.Dense(units=10, activation="softmax")
])

# モデルのコンパイル (最適化アルゴリズム、損失関数の指定)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# トレーニングデータの読み込みと学習
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
model.fit(dataset, epochs=10) 

# 学習済みモデルの保存
model.save("brain_hacking_model.h5")
