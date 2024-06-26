import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import CountVectorizer
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

def main():
    st.title("GDO口コミ分析")

    # ユーザー入力
    url = st.text_input("URLを入力してください:")
    pageNum = st.number_input("ページ数を入力してください:", value=3, step=1)

    if st.button("口コミを分析する"):
        # スクレイピング
        st.write("ちょっとまってね...")
        urlbase = url.rstrip('/')  # URLの末尾のスラッシュを削除
        urlall = [f"{urlbase}?p={i}" for i in range(1, pageNum + 1)]

        data_list = []
        review_hashes = set()

        for url in urlall:
            response = requests.get(url)
            time.sleep(2)  # 2秒の待機時間を設定
            html = response.text
            reviews = parse_reviews(html)
            for review in reviews:
                review_hash = get_review_hash(review)
                if review_hash not in review_hashes:
                    data_list.append(review)
                    review_hashes.add(review_hash)

        # データフレーム作成
        df_all = pd.DataFrame(data_list)
        df_all.loc['平均'] = df_all[['満足度', 'デザイン', 'コスト感', '平均スコア', 'ヘッドスピード', '平均飛距離']].mean()
        
        # CSV出力
        df_all.to_csv('df_all.csv', index=False, encoding='utf-8-sig')
        
        # テキスト読み込み
        file_path = 'df_all.csv'
        column_name = 'レビューコメント'
        texts = read_text_from_csv(file_path, column_name, encoding='utf-8-sig')
        
        # 形態素解析とフレーズ化
        tokenized_texts = tokenize_texts(texts)
        phrases = Phrases(tokenized_texts, min_count=5, threshold=10)
        phraser = Phraser(phrases)
        phrase_texts = phraser[tokenized_texts]
        processed_texts = [' '.join(tokens) for tokens in phrase_texts]
        
        # テキストをベクトル化
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(processed_texts)
        
        # 語彙とその出現頻度を取得
        vocabulary = vectorizer.vocabulary_
        word_frequencies = X.toarray().sum(axis=0)
        
        # 出現頻度の高い順にソートした単語リストを作成
        sorted_vocab = sorted(vocabulary.items(), key=lambda x: word_frequencies[vocabulary[x[0]]], reverse=True)
        
        # 出現回数の少ない順に並べる
        df_most_common = pd.DataFrame({'Word': [word for word, _ in sorted_vocab],
                                       'Frequency': [word_frequencies[vocabulary[word]] for word, _ in sorted_vocab]})
        
        # 上位10件の単語とフレーズを取得
        top_words = df_most_common.head(10)
        
        # 棒グラフの描画
        plt.figure(figsize=(10, 8))
        plt.barh(top_words['Word'], top_words['Frequency'], color='skyblue')
        plt.xlabel('出現回数', fontsize=14)
        plt.ylabel('単語', fontsize=14)
        plt.title('単語の出現回数', fontsize=16)
        plt.tight_layout()
        
        # Streamlitで表示
        st.pyplot(plt)
        
        # df_allにmost_commonを連結
        df_all = pd.read_csv('df_all.csv', encoding='utf-8-sig')
        df_all['most_common_Word'] = df_most_common['Word']
        df_all['most_common_Frequency'] = df_most_common['Frequency']
        
        # 新しいDataFrameをExcelファイルとして出力
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_all.to_excel(writer, index=False, sheet_name='Sheet1')
        
        # 表示
        st.write(df_all)
        
        # エクセルファイルをダウンロードするボタン
        st.markdown(get_excel_download_link(df_all), unsafe_allow_html=True)

def get_excel_download_link(df):
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    excel_data = excel_buffer.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="口コミ分析.xlsx">ダウンロード</a>'
    return href

if __name__ == "__main__":
    main()
