from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
# nltk.download('vader_lexicon')

user_agent = "Scrapper 1.0 by /u/python_engineer"
reddit = praw.Reddit(
    client_id = 'EKYbk2uxVIhb7FqBwArQcA',
    client_secret = 'Eq0AckJSclfNMt9yeII_h7jks60bCQ',
    user_agent=user_agent
)

headlines = set()
for submission in reddit.subreddit('Anime').hot(limit=None):
    # print(submission.title)
    # print(submission.id)
    # print(submission.author)
    # print(submission.created_utc)
    # print(submission.score)
    # print(submission.upvote_ratio)
    # print(submission.url)
    # break
    headlines.add(submission.title)
print(len(headlines))
df = pd.DataFrame(headlines)
df.head()

df.to_csv('headlines.csv', header=False, encoding='utf-8', index=False)

sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)
pprint(results[:3], width=100)

df = pd.DataFrame.from_records(results)
df.head()
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < 0.2, 'label'] = -1
df.head()

df2 = df[['headline', 'label']]
df2.to_csv('reddit_headlines_labels.csv', encoding='utf-8', index=False)
