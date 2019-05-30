#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "2334424681-4Gc0aAl9oLhS0xpQ04HHmZLb62oYz1n3YKT2xw8"
access_token_secret = "aAH6X8SRkgwcPufCNMGmhkhLS0sYJhBpziKFneqyHzP5h"
consumer_key = "gR0dPUOOu4emnnWDP8fyUglZP"
consumer_secret = "k3AXsKLtzM0ATfCp4GYUVJ6wKBg2X25dURS53z3xcNRKY2OtIY"


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
    stream.filter(locations=[-161.75583, 19.50139,-68.01197,64.85694])