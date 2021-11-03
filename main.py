import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import requests
from bs4 import BeautifulSoup

import MeCab
import re

from gensim.models import doc2vec

hicalu = 'https://stand.fm/channels/5f44bb96907968e29d8f3924'
new = "ゴッホが死後に有名になった理由と立役者"

tagger = MeCab.Tagger("-Owakati")

def wakati(sentence):
    # MeCabで分かち書き
    sentence = tagger.parse(sentence)
    # 半角全角英数字除去
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    # 記号もろもろ除去
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    # スペースで区切って形態素の配列へ
    wakati = sentence.split(" ")
    # 空の要素は削除
    wakati = list(filter(("").__ne__, wakati))
    return wakati

class Get_dataset:
    def __init__(self, url):
        self.url = url
    def _get_episodes(self):
        # scraping
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # extract title & url
        title = soup.find_all('div', class_ = "css-901oao css-cens5h r-190imx5 r-a023e6 r-1od2jal r-rjixqe")
        url = soup.find_all('a')
        
        title_list = []
        url_list = []
        for t in title:
            title_name = t.getText()
            title_list.append(title_name)
        for u in url:
            if "episodes" in u.get('href'):
                episode_url = "https://stand.fm" + str(u.get('href'))
                url_list.append(episode_url)
        return (title_list, url_list)
    def _get_master(self):
        data = self._get_episodes()
        title_list, url_list = data[0], data[1]
        episodes_master = pd.DataFrame()
        episodes_master["title"] = title_list
        episodes_master['url'] = url_list
        title_wakati_list = []
        for t in episodes_master['title']:
            title_wakati_list.append(wakati(t))
        return (episodes_master, title_wakati_list)

class Doc2Vec:
    def __init__(self, data, new_title):
        self.episodes_master = data[0]
        self.title_list = data[1]
        self.new_title = new_title
    def train(self):
        trainings = [doc2vec.TaggedDocument(words = data, tags = [i]) for i, data in enumerate(self.title_list)]
        model = doc2vec.Doc2Vec(documents=trainings, dm=1, vector_size=100, window=8, min_count=1, workers=4)
        return model
    def _inference(self):
        model = self.train()
        infe = model.dv.most_similar([model.infer_vector(wakati(self.new_title))])
        return infe
    def _recommend(self):
        rec = self._inference()
        rec_list = []
        for r in rec:
            rec_list.append(*self.episodes_master.query('index=={}'.format(r[0])).values.tolist())
        return rec_list

def main():

    get_data = Get_dataset(hicalu)
    data = get_data._get_master()

    rec_list = Doc2Vec(data, new)._recommend()
    rec_episodes = pd.DataFrame(rec_list,columns=['title','url'])
    rec_episodes.head(3).to_excel('rec_episodes.xlsx', sheet_name="rec_episodes",index=False)

if __name__ == "__main__":
    main()

