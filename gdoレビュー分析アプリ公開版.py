import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
import unicodedata
import re
import japanize_matplotlib
import base64
import io
import hashlib
import xlsxwriter

def fetch_reviews(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_reviews(html):
    soup = BeautifulSoup(html, 'html.parser')
    reviews = soup.find_all('div', class_='reviewContribute')
    data_list = []

    for review in reviews:
        review_data = {}
        stars = review.find('div', class_='recomStar').find_all('img')
        review_data['満足度'] = int(stars[0]['alt'].replace('★', ''))
        review_data['デザイン'] = int(stars[1]['alt'].replace('★', ''))
        review_data['コスト感'] = int(stars[2]['alt'].replace('★', ''))
        user_info = review.find('dl', class_='reviewUser').find('dd').text.strip()
        review_data['性別'] = re.findall(r'（(.*?)）', user_info)[0]
        height_match = re.findall(r'身長：(\d+)cm', user_info)
        review_data['身長'] = int(height_match[0]) if height_match else None
        weight_match = re.findall(r'体重：(\d+)kg', user_info)
        review_data['体重'] = int(weight_match[0]) if weight_match else None
        review_data['レビューコメント'] = unicodedata.normalize('NFKC', review.find('dl', class_='reviewArticle').find('dd').text.strip()).lower()
        shaft_info = review.find_all('li')
        for info in shaft_info:
            if 'シャフト：' in info.text:
                review_data['シャフト'] = unicodedata.normalize('NFKC', info.text.split('：')[1].strip()).lower()
            elif 'シャフトフレックス：' in info.text:
                review_data['シャフトフレックス'] = unicodedata.normalize('NFKC', info.text.split('：')[1].strip()).lower()
        golfer_info = review.find_all('li')
        for info in golfer_info:
            if 'ゴルファータイプ：' in info.text:
                review_data['ゴルファータイプ'] = unicodedata.normalize('NFKC', info.text.split('：')[1].strip()).lower()
            elif '平均スコア：' in info.text:
                scores = re.findall(r'\d+', info.text)
                if scores:
                    review_data['平均スコア'] = sum(map(int, scores)) / len(scores)
            elif 'ヘッドスピード：' in info.text:
                speeds = re.findall(r'\d+', info.text)
                if speeds:
                    review_data['ヘッドスピード'] = sum(map(int, speeds)) / len(speeds)
            elif '平均飛距離：' in info.text:
                distances = re.findall(r'\d+', info.text)
                if distances:
                    review_data['平均飛距離'] = sum(map(int, distances)) / len(distances)
            elif '持ち球：' in info.text:
                review_data['持ち球'] = unicodedata.normalize('NFKC', info.text.split('：')[1].strip()).lower()
            elif '弾道高さ：' in info.text:
                review_data['弾道高さ'] = unicodedata.normalize('NFKC', info.text.split('：')[1].strip()).lower()
        data_list.append(review_data)

    return data_list

def get_review_hash(review_data):
    review_str = ''.join(str(value) for value in review_data.values())
    return hashlib.md5(review_str.encode()).hexdigest()

def read_text_from_csv(file_path, column_name, encoding='utf-8'):
    df = pd.read_csv(file_path, encoding=encoding)
    df[column_name] = df[column_name].fillna('')
    return df[column_name].tolist()

def tokenize_texts(texts):
    t = Tokenizer()
    token_filters = [POSKeepFilter(['名詞'])]
    analyzer = Analyzer(tokenizer=t, token_filters=token_filters)
    return [[token.surface for token in analyzer.analyze(unicodedata.normalize('NFKC', text))] for text in texts]

def get_excel_download_link(buffer, filename):
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">⬇️ {filename} をダウンロード</a>'

def main():
    st.title("GDO口コミ分析ツール")

    url = st.text_input("URLを入力してください:")
    model_name = st.text_input("モデル名を入力してください（例：XXIO12）")

    if st.button("口コミを分析する"):
        with st.spinner("口コミ収集中..."):
            urlbase = url.rstrip('/')
            page = 1
            data_list = []
            review_hashes = set()

            while True:
                full_url = f"{urlbase}?p={page}"
                response = requests.get(full_url)
                time.sleep(2)
                html = response.text
                reviews = parse_reviews(html)

                if not reviews:
                    break

                for review in reviews:
                    review_hash = get_review_hash(review)
                    if review_hash not in review_hashes:
                        data_list.append(review)
                        review_hashes.add(review_hash)

                page += 1

        if not data_list:
            st.error("口コミが見つかりませんでした。")
            return

        df_all = pd.DataFrame(data_list)
        df_all.loc['平均'] = df_all[['満足度', 'デザイン', 'コスト感', '平均スコア', 'ヘッドスピード', '平均飛距離']].mean()

        texts = df_all['レビューコメント'].fillna('').tolist()
        tokenized_texts = tokenize_texts(texts)
        phrases = Phrases(tokenized_texts, min_count=5, threshold=10)
        phraser = Phraser(phrases)
        phrase_texts = phraser[tokenized_texts]
        processed_texts = [' '.join(tokens) for tokens in phrase_texts]

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(processed_texts)
        vocabulary = vectorizer.vocabulary_
        word_frequencies = X.toarray().sum(axis=0)

        sorted_vocab = sorted(vocabulary.items(), key=lambda x: word_frequencies[vocabulary[x[0]]], reverse=True)
        df_most_common = pd.DataFrame({
            'Word': [word for word, _ in sorted_vocab],
            'Frequency': [word_frequencies[vocabulary[word]] for word, _ in sorted_vocab]
        })
        top_words = df_most_common.head(10)

        plt.figure(figsize=(10, 8))
        plt.barh(top_words['Word'], top_words['Frequency'], color='skyblue')
        plt.xlabel('出現回数', fontsize=14)
        plt.ylabel('単語', fontsize=14)
        plt.title('単語の出現回数', fontsize=16)
        plt.tight_layout()
        st.pyplot(plt)

        today = datetime.now().strftime('%Y%m%d')
        filename = f"{model_name}_{today}.xlsx"

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_all.to_excel(writer, index=False, sheet_name='Sheet1')

        st.write(df_all)
        st.markdown(get_excel_download_link(excel_buffer, filename), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
