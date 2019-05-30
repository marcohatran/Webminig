#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "1012675774313218049-C2inpAwtWRLNiKSbXLwyLqWHlyIX7i"
access_token_secret = "Y9w8c1lE7aj19CFzqx9aGYcTdFehN4BIvIaZyoPDskhMl"
consumer_key = "LxulRXUWs3wGOkoFc4DYSmB4s"
consumer_secret = "FqCeyhMlShkvMFVBdlFmq9sYu2eWGFQDJLMwtbfAXUy8CYGryU"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(locations=[-122.75,36.8,-121.75,37.8,-74,40,-73,41,-99,31,-97,33])