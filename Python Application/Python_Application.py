import pandas as pd
import texthero as hero
from texthero import preprocessing

df=pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")

df['pca'] = (
            df['text']
            .pipe(hero.clean)
            .pipe(hero.tfidf)
            .pipe(hero.pca)
   )

hero.scatterplot(df, col='pca', color='topic', title="PCA BBC Sports News")

NUM_TOP_WORDS=5
df.groupby('topic')['text'].apply(lambda x: hero.top_words(x)[:NUM_TOP_WORDS])