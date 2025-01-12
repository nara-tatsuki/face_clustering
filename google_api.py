# -*- coding:utf-8 -*-
import httplib2
import os
import hashlib
from googleapiclient.discovery import build
import time


def fetch_image_urls(api_key, cse_key, search_word):
    """Google Custom Search APIを使用して画像URLを取得する"""
    service = build("customsearch", "v1", developerKey=api_key)
    response_list = []
    img_urls = []

    start_index = 1
    for page_num in range(15):
        print(f"Fetching page number: {page_num + 1}")
        try:
            response = service.cse().list(
                q=search_word,     # 検索ワード
                cx=cse_key,        # カスタム検索エンジンキー
                lr='lang_ja',      # 言語設定
                num=10,            # 一度のリクエストで取得する画像の数（最大10）
                start=start_index,
                searchType='image' # 画像検索
            ).execute()

            response_list.append(response)
            start_index = response.get("queries").get("nextPage")[0].get("startIndex")

        except Exception as e:
            print(f"Error fetching page {page_num + 1}: {e}")
            break

    # 取得した画像のURLをリストに追加
    for response in response_list:
        items = response.get('items', [])
        for item in items:
            img_urls.append(item['link'])

    return img_urls


def download_images(save_dir_path, img_urls):
    """指定された画像URLから画像をダウンロードして保存する"""
    os.makedirs(save_dir_path, exist_ok=True)
    http = httplib2.Http(".cache")

    for img_url in img_urls:
        try:
            extension = os.path.splitext(img_url)[-1].lower()
            if extension in {'.jpg', '.jpeg', '.gif', '.png', '.bmp'}:
                # 画像URLをハッシュ化してファイル名に使用
                encoded_url = img_url.encode('utf-8')
                hashed_url = hashlib.sha3_256(encoded_url).hexdigest()
                img_path = os.path.join(save_dir_path, f"{hashed_url}{extension}")

                # 画像をダウンロードして保存
                response, content = http.request(img_url)
                if response.status == 200:
                    with open(img_path, 'wb') as img_file:
                        img_file.write(content)
                    print(f"Saved image: {img_url}")

        except Exception as e:
            print(f"Failed to download {img_url}: {e}")
            continue

def google_api(API_KEY, CUSTOM_SEARCH_ENGINE, search_word, dir_name):
    # 画像URLを取得し、画像をダウンロード
    img_urls = fetch_image_urls(API_KEY, CUSTOM_SEARCH_ENGINE, search_word)
    download_images(dir_name, img_urls)