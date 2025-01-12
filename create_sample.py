import os
import random
import pathlib
import shutil
import glob
import cv2
import numpy as np

def load_images(input_dir_trim):
    image_info_lists = []
    # 指定したパスパターンに一致するファイルの取得
    image_paths = glob.glob(input_dir_trim)
    # ファイルごとの読み込み
    for image_path in image_paths:
        path = pathlib.Path(image_path)
        fullpath = str(path.resolve())
        filename = path.name
        # 画像読み込み
        image = cv2.imread(fullpath)
        if image is None:
            print(f"画像ファイル[{fullpath}]を読み込めません")
            continue
        image_info_lists.append((filename, image))
        print(f"画像ファイル[{fullpath}]を読み込みました")
    return image_info_lists

def detect_image_face(file_path, image, cascade_filepath):
    # 画像ファイルのグレースケール化
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # カスケードファイルの読み込み
    cascade = cv2.CascadeClassifier(cascade_filepath)
    # 顔認識
    faces = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=15, minSize=(64, 64))
    if len(faces) == 0:
        print(f"顔認識失敗")
        return
    # 1つ以上の顔を認識
    face_count = 1
    for (xpos, ypos, width, height) in faces:
        face_image = image[ypos:ypos+height, xpos:xpos+width]
        if face_image.shape[0] > 64:
            face_image = cv2.resize(face_image, (64, 64))
        # 保存
        path = pathlib.Path(file_path)
        directory = str(path.parent.resolve())
        filename = path.stem
        extension = path.suffix
        output_path = os.path.join(directory, f"{filename}_{face_count:03}{extension}")
        print(f"出力ファイル（絶対パス）:{output_path}")
        cv2.imwrite(output_path, face_image)
        face_count = face_count + 1

def scratch_image(image, use_flip=True, use_threshold=True, use_filter=True):
    # どの水増手法を利用するか（フリップ、閾値、平滑化）
    methods = [use_flip, use_threshold, use_filter]
    # ぼかしに使うフィルターの作成
    # filter1 = np.ones((3, 3))
    # オリジナルの画像を配列に格納
    images = [image]
    # 水増手法の関数
    scratch = np.array([
        # フリップ処理
        lambda x: cv2.flip(x, 1),
        # 閾値処理
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        # 平滑化処理
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    ])
    # 画像の水増
    doubling_images = lambda f, img: np.r_[img, [f(i) for i in img]]
    for func in scratch[methods]:
        images = doubling_images(func, images)
    return images

# 画像の切り抜き
def execute_face_trim(input_dir_trim,output_dir_trim):
    print("===================================================================")
    print("イメージ顔認識 OpenCV 利用版")
    print("指定した画像ファイルの正面顔を認識して抜き出し、サイズ変更64x64を行います。")
    print("===================================================================")

    cascade_filepath = r"C:\Users\htt06\programing\python\face_clustering\haarcascade_frontalface_default.xml"
    # ディレクトリの作成
    if not os.path.isdir(output_dir_trim):
        os.mkdir(output_dir_trim)

    # 画像ファイルの読み込み
    image_info_lists = load_images(input_dir_trim)

    # 画像ごとの顔認識
    for image_info in image_info_lists:
        file_path = os.path.join(output_dir_trim, f"{image_info[0]}")
        image = image_info[1]
        detect_image_face(file_path, image, cascade_filepath)
    delete_path = os.path.dirname(input_dir_trim)
    shutil.rmtree(delete_path)

# 画像の水増し
def execute_scratch_image(input_dir_scratch,output_dir_test,output_dir_scratch):
    print("===================================================================")
    print("イメージ水増し OpenCV 利用版")
    print("指定した画像ファイルの水増し（フリップ＋閾値＋平滑化で8倍）を行います。")
    print("===================================================================")
    # ディレクトリの作成
    if not os.path.isdir(output_dir_test):
        os.mkdir(output_dir_test)

    # ディレクトリの作成
    if not os.path.isdir(output_dir_scratch):
        os.mkdir(output_dir_scratch)

    # 対象画像のうち2割をテスト用として退避
    image_files = glob.glob(input_dir_scratch)
    random.shuffle(image_files)
    for i in range(len(image_files)//5):
        shutil.move(str(image_files[i]), output_dir_test)

    # 画像ファイルの読み込み
    name_images = load_images(input_dir_scratch)

    # 画像ごとの水増し
    for name_image in name_images:
        filename, extension = os.path.splitext(name_image[0])
        image = name_image[1]
        # 画像の水増し
        scratch_face_images = scratch_image(image)
        # 画像の保存
        for idx, image in enumerate(scratch_face_images):
            output_path = os.path.join(output_dir_scratch, f"{filename}_{str(idx)}{extension}")
            print(f"出力ファイル（絶対パス）:{output_path}")
            cv2.imwrite(output_path, image)
    delete_path = os.path.dirname(input_dir_scratch)
    shutil.rmtree(delete_path)

# フォルダ名を連番で置換 移動
def rename_images(directory, new_dir, prefix, extension):
    # ディレクトリ内のファイル一覧を取得
    files = os.listdir(directory)
    
    # 画像ファイルを番号順にリネーム
    for i, filename in enumerate(sorted(files)):
        # ファイルの拡張子を確認
        file_ext = os.path.splitext(filename)[1].lower()  # .jpg, .jpeg, .png など
        if file_ext in [".jpg", ".jpeg", ".png"]:  # 対応する画像形式をここに追加
            # 新しいファイル名を作成
            new_name = f"{prefix}{i+1:03}{extension}"  # 001, 002 など
            src = os.path.join(directory, filename)
            dst = os.path.join(new_dir, new_name)

            # ファイル名の変更
            os.rename(src, dst)
            print(f"Renamed: {src} -> {dst}")
    shutil.rmtree(directory)