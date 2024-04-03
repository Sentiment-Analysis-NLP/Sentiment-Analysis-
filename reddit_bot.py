import praw

username = 'No_Hall_664'
# password = input("Enter your Reddit password: ")
password = 'aman06062003'
client_id = 'tghQei_hJGceWu0dnzTfWg'
client_secret = 'Gpomlr4ozb362AHzi4zYC93KwQSmhg'

reddit_instance = praw.Reddit(
    client_id = client_id,
    client_secret = client_secret,
    username = username,
    password = password,
    
    user_agent = 'test_bot'
)

# print(reddit_instance.user.me())
