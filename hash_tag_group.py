import json
import pandas as pd
# import matplotlib.pyplot as plt
import time

# return 2 array: 
# hashtags = [hashtag_name, ...]
# tweet_group = [[tweets_of_hashtag_i], ...]
def get_hash_tag_group(tweets):
    hashtags = ["#no_hashtag"]
    
    for tweet in tweets:
        for hashtag in tweet['entities']['hashtags']:
            if hashtag['text'] not in hashtags:
                hashtags.append(hashtag['text'])
                
    tweets_of_hashtag = [[] for i in range(len(hashtags))]
    
    for tweet in tweets:
        hashtags_of_tweet = get_hashtags_of_tweet(tweet)

        # no_hashtag
        if len(hashtags_of_tweet) == 0:
            tweets_of_hashtag[0].append(tweet)

        else:
            for hashtag in hashtags_of_tweet:
                index = hashtags.index(hashtag)
                tweets_of_hashtag[index].append(tweet)


    return [hashtags, tweets_of_hashtag]



# Return [[hashtags, tweets_of_hashtag], ...]
def hash_tag_group():
    tweets_data_path = './twitter_data2.txt'

    # Covert data from json file
    tweets_data = []
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue
    
    # print("Length: " + str(len(tweets_data)))
    
    # get hash tag group
    hash_tag_group= []

    tweets_every_10_minutes = []
    flag = True
    for tweet in tweets_data:
        # Divide into groups every 10 minutes
        if int(tweet['created_at'][14:16])%10 == 0 and flag == True:
            flag = False
            new_group = get_hash_tag_group(tweets_every_10_minutes)
            hash_tag_group.append(new_group)

            # continue in next 10 minutes
            tweets_every_10_minutes = []
            tweets_every_10_minutes.append(tweet)
        else:
            tweets_every_10_minutes.append(tweet)
            if int(tweet['created_at'][14:16])%10 != 0:
                flag = True
            
    return hash_tag_group

def get_hashtags_of_tweet(tweet):
    hashtags = []
    for hashtag in tweet['entities']['hashtags']:
        if hashtag['text'] not in hashtags:
            hashtags.append(hashtag['text'])
    
    return hashtags

if __name__ == '__main__':
    # print some result examples
    for j in range(100):
        group = hash_tag_group()[j]
        hashtags = group[0]
        tweets_of_hashtag = group[1]
        for i in range(len(hashtags)):
            print("#########################")
            print("#########################")
            print("Name: " + hashtags[i])
            print("Length: " + str(len(tweets_of_hashtag[i])))
            print("TWEET: ")
            print(get_hashtags_of_tweet(tweets_of_hashtag[i][0]))