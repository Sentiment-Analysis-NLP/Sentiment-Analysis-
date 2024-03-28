import praw
import pandas as pd

username = ''
password = ''
client_id = ''
client_secret = ''

reddit_instance = praw.Reddit(
    client_id = client_id,
    client_secret = client_secret,
    username = username,
    password = password,
    
    user_agent = 'test_bot'
)

# print(reddit_instance.user.me())

subreddit = reddit_instance.subreddit("IndianStockMarket")
print(subreddit)

top_25_sub = subreddit.top(limit=25, time_filter='week')


data = []
for sub in top_25_sub:
    print(sub.title)
    print(sub.selftext)
    data.append([sub.title, sub.selftext])
    # sub.comment_sort = "confidence"  #

dataset = pd.DataFrame(data, columns=['Title', 'Content'])

dataset.to_csv("data.csv", index=False)