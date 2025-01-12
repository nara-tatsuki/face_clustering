import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from tensorflow.keras.utils import plot_model


# データ読み込み
def load_images(image_directory):
    image_file_list = []
    image_file_name_list = os.listdir(image_directory)
    print(f"対象画像ファイル数：{len(image_file_name_list)}")
    for image_file_name in image_file_name_list:
        image_file_path = os.path.join(image_directory, image_file_name)
        image = cv2.imread(image_file_path)
        if image is None:
            print(f"画像ファイル[{image_file_name}]を読み込めません")
            continue
        image_file_list.append((image_file_name, image))
    return image_file_list


# ラベル付け
def labeling_images(image_file_list):
    labels = {"cute_soft": 0, "cute_hard": 1, "fresh_soft": 2, "fresh_hard": 3, "elegant_soft": 4, "elegant_hard": 5, "cool_soft": 6, "cool_hard": 7}
    x_data, y_data = [], []
    for idx, (file_name, image) in enumerate(image_file_list):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x_data.append(image)
        label = -1
        for key, value in labels.items():
            if key in file_name.lower():
                label = value
                break
        if label == -1:
            raise ValueError(f"ファイル名に有効なタイプ名が含まれていません: {file_name}")
        y_data.append(label)
    return np.array(x_data), np.array(y_data)


# データリサイズ
def preprocess_images(image_list, target_size=(64, 64)):
    resized_images = [cv2.resize(image, target_size) for image in image_list]
    return np.array(resized_images)


# ディレクトリ削除
def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)


# モデル構築
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# メイン処理
def main():
    print("===================================================================")
    print("8タイプモデル学習")
    print("===================================================================")

    TRAIN_IMAGE_DIR = "./model_image"
    TEST_IMAGE_DIR = "./test_image"
    OUTPUT_MODEL_DIR = "./model"
    OUTPUT_MODEL_FILE = "model.h5"
    OUTPUT_PLOT_FILE = "model.png"

    num_classes = 8
    batch_size = 32
    epochs = 20
    target_size = (64, 64)

    # 出力ディレクトリの準備
    if not os.path.isdir(OUTPUT_MODEL_DIR):
        os.mkdir(OUTPUT_MODEL_DIR)
    delete_dir(OUTPUT_MODEL_DIR, False)

    # データ読み込みと前処理
    train_file_list = load_images(TRAIN_IMAGE_DIR)
    x_train, y_train = labeling_images(train_file_list)
    x_train = preprocess_images(x_train, target_size)
    y_train = to_categorical(y_train, num_classes)

    test_file_list = load_images(TEST_IMAGE_DIR)
    x_test, y_test = labeling_images(test_file_list)
    x_test = preprocess_images(x_test, target_size)
    y_test = to_categorical(y_test, num_classes)

    print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
    print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")

    # モデル作成
    model = build_model(input_shape=(target_size[0], target_size[1], 3), num_classes=num_classes)
    model.summary()

    # モデル学習
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # モデル保存と可視化
    model.save(os.path.join(OUTPUT_MODEL_DIR, OUTPUT_MODEL_FILE))
    plot_model(model, to_file=os.path.join(OUTPUT_MODEL_DIR, OUTPUT_PLOT_FILE), show_shapes=True)

    # 学習結果の可視化
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # モデル評価
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")


if __name__ == "__main__":
    main()
