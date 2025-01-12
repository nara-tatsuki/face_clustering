import os
from google_api import google_api
from create_sample import execute_face_trim,execute_scratch_image,rename_images

from dotenv import load_dotenv

load_dotenv()

# APIキーをenvファイルから取得
API_KEY = os.getenv('GOOGLE_API_KEY')
CUSTOM_SEARCH_ENGINE = os.getenv('CUSTOM_SEARCH_ENGINE')

# 学習させる芸能人の名前
search_word = '赤西仁'

# その芸能人の顔タイプ
dir_name = 'cool'

CURRENT_PATH = r'C:\Users\htt06\programing\python\face_clustering'
INPUT_DIR_TRIM = os.path.join(CURRENT_PATH ,dir_name + "\*")
OUTPUT_DIR_TRIM = os.path.join(CURRENT_PATH,"trim_" + dir_name)
INPUT_DIR_SCRATCH = OUTPUT_DIR_TRIM + "\*"
OUTPUT_DIR_TEST = os.path.join(CURRENT_PATH,"test_" + dir_name)
OUTPUT_DIR_SCRATCH = os.path.join(CURRENT_PATH,"scratch_" + dir_name)
TEST_DIR = os.path.join(CURRENT_PATH,"test_image")
MODEL_DIR = os.path.join(CURRENT_PATH,"model_image")

# googleAPIで画像収集
google_api(API_KEY, CUSTOM_SEARCH_ENGINE, search_word,dir_name)

# 顔画像切り取り
execute_face_trim(INPUT_DIR_TRIM,OUTPUT_DIR_TRIM)

# 画像水増し
execute_scratch_image(INPUT_DIR_SCRATCH,OUTPUT_DIR_TEST,OUTPUT_DIR_SCRATCH)

# ファイル名変更
rename_images(OUTPUT_DIR_SCRATCH,MODEL_DIR,dir_name + "_", ".jpg")
rename_images(OUTPUT_DIR_TEST,TEST_DIR,dir_name + "_", ".jpg")