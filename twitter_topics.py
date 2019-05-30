#!/usr/bin/env python
import re
import codecs
import sys
import nltk
import numpy as np
from datetime import datetime
from sklearn import preprocessing
import fastcluster
from collections import Counter
import scipy.cluster.hierarchy as sch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances


def load_stopwords():
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(['this', 'that', 'the', 'might', 'have', 'been', 'from',
                       'they', 'will', 'has', 'having', 'had', 'how',
                       'were', 'why', 'and', 'his', 'her', 'was', 'its', 'per', 'cent',
                       'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among',
                       'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'by',
                       'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever',
                       'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his',
                       'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let',
                       'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'nor',
                       'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said',
                       'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their',
                       'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
                       'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who',
                       'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your', 've', 're', 'rt', 'retweet', '#fuckem', '#fuck',
                       'fuck', 'ya', 'yall', 'yay', 'youre', 'youve', 'ass', 'factbox', 'com', '&lt', 'th',
                       'retweeting', 'dick', 'fuckin', 'shit', 'via', 'fucking', 'shocker', 'wtf', 'hey', 'ooh', 'rt&amp', '&amp',
                       '#retweet', 'retweet', 'goooooooooo', 'hellooo', 'gooo', 'fucks', 'fucka', 'bitch', 'wey', 'sooo', 'helloooooo', 'lol', 'smfh'])
    # print(stop_words)
    stop_words = set(stop_words)
    return stop_words

def spam_tweet(text):
    if 'Jordan Bahrain Morocco Syria Qatar Oman Iraq Egypt United States' in text:
        return True
		
    if 'Some of you on my facebook are asking if it\'s me' in text:
        return True
		
    if 'Construction' in text:
        return True
	
    if 'follow me please' in text:
        return True
	
    if 'please follow me' in text:
        return True		
    if 'Incident' in text:
        return True
    return False	

def normalize_text(text):
    try:
        text = text.decode('utf-8')
    except:
        pass
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))', '', text)
    text = re.sub('@[^\s]+', '', text)
    text = re.sub('#([^\s]+)', '', text)
    text = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]', ' ', text)
    text = re.sub('[\d]', '', text)
    text = text.replace(".", '')
    text = text.replace("'", ' ')
    text = text.replace("\"", ' ')
    # text = text.replace("-", " ")
    # normalize some utf8 encoding
    text = text.replace("&gt", ' ').replace("\x8c", ' ')
    text = text.replace("\xa0", ' ')
    text = text.replace("\x9d\x92", ' ').replace("\x9a\xaa\xf0\x9f\x94\xb5", ' ').replace(
        "\xf0\x9f\x91\x8d\x87\xba\xf0\x9f\x87\xb8", ' ').replace("\x9f", ' ').replace("\x91\x8d", ' ')
    text = text.replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8", ' ').replace("\xf0", ' ').replace(
        '\xf0x9f', '').replace("\x9f\x91\x8d", ' ').replace("\x87\xba\x87\xb8", ' ')
    text = text.replace("\xe2\x80\x94", ' ').replace("\x9d\xa4", ' ').replace("\x96\x91", ' ').replace(
        "\xe1\x91\xac\xc9\x8c\xce\x90\xc8\xbb\xef\xbb\x89\xd4\xbc\xef\xbb\x89\xc5\xa0\xc5\xa0\xc2\xb8", ' ')
    text = text.replace("\xe2\x80\x99s", " ").replace("\xe2\x80\x98", ' ').replace(
        "\xe2\x80\x99", ' ').replace("\xe2\x80\x9c", " ").replace("\xe2\x80\x9d", " ")
    text = text.replace("\xe2\x82\xac", " ").replace("\xc2\xa3", " ").replace("\xc2\xa0", " ").replace(
        "\xc2\xab", " ").replace("\xf0\x9f\x94\xb4", " ").replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f", "")
    return text

def custom_tokenize_text(text):
    REGEX = re.compile(r",\s*")
    tokens = []
    for tok in REGEX.split(text):
        #if "@" not in tok and "#" not in tok:
        if "@" not in tok:
            #tokens.append(stem(tok.strip().lower()))
            tokens.append(tok.strip().lower())
    return tokens

def nltk_tokenize(text):
    tokens = []
    pos_tokens = []
    entities = []
    features = []
    try:
        tokens = text.split()
        for word in tokens:
            if word.lower() not in stop_words and len(word) > 1:
                features.append(word)
    except:
        pass
    return [tokens, pos_tokens, entities, features]


def process_json_tweet(text):
    features = []

    if len(text.strip()) == 0:
        return []
    text = normalize_text(text)
    try:
        [tokens, pos_tokens, entities, features] = nltk_tokenize(text)
    except:
        print("nltk tokenize+pos pb!")
    return features


if __name__ == "__main__":

    # Open file and get time for each window
    file_timeordered_tweets = codecs.open(sys.argv[1], 'r', 'utf-8')
    time_window_mins = float(sys.argv[2])
    stop_words = load_stopwords()
    # fout = codecs.open(sys.argv[3], 'w', 'utf-8')

    # tweet_unixtime_old: Start time of each window. Set = -1 when initial
    # window_corpus: list of words after tokenizer all tweets inside each window
    # tid_to_urls_window_corpus: list urls of each tweets
    # tids_window_corpus: list ids of each tweets
    # tfVocTimeWindows: tf of vocabulary word (voc) for each 4th windows(hourly)
    tweet_unixtime_old = -1
    window_corpus = []
    tid_to_urls_window_corpus = {}
    tids_window_corpus = []
    tid_to_raw_tweet = {}
    tfVocTimeWindows = {}
    ntweets = 0
    t = 0

    # read each line in file after extract twitter.json to text
    for line in file_timeordered_tweets:
        # get information of each line tweet
        [tweet_unixtime, tweet_gmttime, tweet_id, text, hashtags, users, urls, media_urls, nfollowers, nfriends] = eval(line)
        if(spam_tweet(text)):
            continue
        # set start time
        if tweet_unixtime_old == -1:
            tweet_unixtime_old = tweet_unixtime
        # if tweet's time inside time window
        if (tweet_unixtime - tweet_unixtime_old) < time_window_mins * 60:
            ntweets += 1

            # tokenizer tweet
            features = process_json_tweet(text)
            tweet_bag = ""
            
            try:
                for user in set(users):
                    tweet_bag += "@" + user.lower() + ","
                for tag in set(hashtags):
                    if tag.lower() not in stop_words:
                        tweet_bag += "#" + tag.lower() + ","
                for feature in features:
                    tweet_bag += feature + ","
            except:
                pass

            # append word, user and hashtag to window_corpus
            if len(users) < 4 and len(hashtags) < 4 and len(features) > 3 and len(tweet_bag.split(",")) > 4 and not str(features).upper() == str(features):
                tweet_bag = tweet_bag[:-1]
                window_corpus.append(tweet_bag)
                tids_window_corpus.append(tweet_id)
                tid_to_urls_window_corpus[tweet_id] = media_urls
                tid_to_raw_tweet[tweet_id] = text
            
        # finish time_window 
        else:
            # get time
            dtime = datetime.fromtimestamp(tweet_unixtime_old).strftime("%d-%m-%Y %H:%M")
            print("-----------------------------------------------------------")
            print ("\nWindow Starts GMT Time:", dtime, "\n")
            # print(window_corpus)
            tweet_unixtime_old = tweet_unixtime	
            t += 1
            
            # vectorize window_corpus with ngram(1,2)
            vectorizer = TfidfVectorizer(tokenizer=custom_tokenize_text, binary=True, min_df=2, ngram_range=(1,2))
            X = vectorizer.fit_transform(window_corpus)
            # print(window_corpus)
            map_index_after_cleaning = {}
            Xclean = np.zeros((1, X.shape[1]))
            for i in range(0, X.shape[0]):
                # keep sample with size at least 5 word in tweet
                # if X[i].sum() > 4:
                Xclean = np.vstack([Xclean, X[i].toarray()])
                map_index_after_cleaning[Xclean.shape[0] - 2] = i

            Xclean = Xclean[1:,]
            X = Xclean
            # print(X.shape)
            Xdense = np.matrix(X).astype('float')
            # X_scaled = preprocessing.scale(Xdense)
            # X_normalized = preprocessing.normalize(X_scaled, norm='l2')
            
            vocX = vectorizer.get_feature_names()
            # print(vocX)

            # boost weight for hashtag when calculate score tf of Vocabulary(Voc)
            boost_entity = {}
            for voc in vocX:
                if "#" in str(voc):
                    boost_entity[voc] = 1.25
                else:
                    boost_entity[voc] = 1.0

            tfX = X.sum(axis=0)
            #print "tfX:", tfX
            tfVoc = {}
            idfVoc = {}
            wtfVoc = {}
            boosted_wtfVoc = {}	
            keys = vocX
            vals = tfX
            # dictionary tfVoc(key,val): key - word of Voc, val - appear frequent of this key
            for k,v in zip(keys, vals):
                # print(k, "  ", v)
                tfVoc[k] = v
                idfVoc[k] = X.shape[0] / (v + 1)
            # calculate score for each word in Voc of current window(higher weight) and previous windows(lower weight)
            max_frequency = tfVoc[max(tfVoc, key=tfVoc.get)]
            # print(sorted( ((v,k) for k,v in idfVoc.items()), reverse=True))
            for k in tfVoc:
                try:
                    tfVocTimeWindows[k] += tfVoc[k]
                    avgtfVoc = (tfVocTimeWindows[k] - tfVoc[k])/(t - 1)
                    #avgtfVoc = (tfVocTimeWindows[k] - tfVoc[k])
                except:
                    tfVocTimeWindows[k] = tfVoc[k]
                    avgtfVoc = 0
                wtfVoc[k] = ((tfVoc[k] + 1) * (np.log(idfVoc[k] + 1))) / ((np.log(avgtfVoc + 1) + 1) * (max_frequency + 1))
                try:
                    boosted_wtfVoc[k] = wtfVoc[k] * boost_entity[k]
                except:
                    boosted_wtfVoc[k] = wtfVoc[k]
            
            # sort dict tfVoc by value score
            # print ("sorted wtfVoc*boost_entity:")
            # print (sorted( ((v,k) for k,v in boosted_wtfVoc.items()), reverse=True))
            # print(len(boosted_wtfVoc))

            # calcluate distance among tweet by cosine distance
            distMatrix = pairwise_distances(Xdense, metric='euclidean')
            # print(distMatrix)
            # print ("fastcluster, average, euclidean")
            # hierarchical clustering and cut dendrogram by 0.5 distance threshold
            L = fastcluster.linkage(distMatrix, method='average')
            dt = 2.0
            # print ("hclust cut threshold:", dt)
            indL = sch.fcluster(L, dt*distMatrix.max(), 'distance')
            freqTwCl = Counter(indL)
            # print ("n_clusters:", len(freqTwCl))
            # print(freqTwCl)

            npindL = np.array(indL)
            freq_th = max(3, int(X.shape[0]*0.0025))
            cluster_score = {}
            # score_tweet = {}
            for clfreq in freqTwCl.most_common(50):
                cl = clfreq[0]
                freq = clfreq[1]
                cluster_score[cl] = 0
                # only get cluster have frequent appear higher than frequent threshold
                if freq >= freq_th:
                    clidx = (npindL == cl).nonzero()[0].tolist()
                    cluster_centroid = X[clidx].sum(axis=0)
                    # print("center ", cluster_centroid.shape)
                    try:
                        cluster_tweet = vectorizer.inverse_transform(cluster_centroid)
                        # print("ttt ", cluster_tweet)
                        for term in np.nditer(cluster_tweet):
                            try:
                                # cluster_score[cl] = max(cluster_score[cl], boosted_wtfVoc[str(term).strip()])
                                # print(term)
                                cluster_score[cl] += boosted_wtfVoc[str(term).strip()]
                            except: pass
                    except: pass
                    # print("cscs, ", cluster_score)
                    cluster_score[cl] /= freq
                else: break
            sorted_clusters = sorted( ((v,k) for k,v in cluster_score.items()), reverse=True)
            # print ("sorted cluster_score:")
            # print (sorted_clusters)
            # print(cluster_score)
            # print(npindL)
            # Get top 20 cluster and select the first tweet as headline for deteted topic
            ntopics = 20
            headline_corpus = []
            orig_headline_corpus = []
            tweet_cluster = {}
            headline_to_cluster = {}
            headline_to_tid = {}
            cluster_to_tids = {}
            final_clusters = {}
            count1 = 0
            count2 = 0
            for score,cl in sorted_clusters[:ntopics]:
                tweet_score = {}
                clidx = (npindL == cl).nonzero()[0].tolist()
                
                # print(clidx)
                # print(score)
                print("##################---------------------#####################")
                for cc in clidx:
                    first_idx = map_index_after_cleaning[cc]
                    keywords = window_corpus[first_idx]
                    keyword = keywords.split(',')
                    tweet_score[cc] = 0
                    
                    print(keywords)
                    for term in keyword:
                        if(not term in boosted_wtfVoc):
                            tweet_score[cc] += 0
                        else:
                            tweet_score[cc] += boosted_wtfVoc[str(term.lower()).strip()]
                        
                
                tweet_score_sort = sorted( ((v,k) for k,v in tweet_score.items()), reverse=True)
                # print(tweet_score_sort)
                first_idx = map_index_after_cleaning[tweet_score_sort[0][1]]
                keywords = window_corpus[first_idx]
                
                orig_headline_corpus.append(keywords)
                headline = ''
                for k in keywords.split(","):
                    if not '@' in k and not '#' in k:
                        headline += k + ","
                
                headline_corpus.append(headline[:-1])
                headline_to_cluster[headline[:-1]] = cl
                # get headline through tweet id
                headline_to_tid[headline[:-1]] = tids_window_corpus[first_idx]
                tids = []
                for i in clidx:
                    idx = map_index_after_cleaning[i]
                    tids.append(tids_window_corpus[idx])
                
                cluster_to_tids[cl] = tids
                tweet_cluster[count1] = tids
                count1 = count1 +1
            print("####################", headline_corpus)
            # reclustering headline to avoid topic fragment
            headline_vectorizer = TfidfVectorizer(tokenizer=custom_tokenize_text, binary=True, min_df=1, ngram_range=(1,1))
            H = headline_vectorizer.fit_transform(headline_corpus)
            # print ("H.shape:", H.shape)
            vocH = headline_vectorizer.get_feature_names()
            Hdense = np.matrix(H.todense()).astype('float')
            distH = pairwise_distances(Hdense, metric='euclidean')
            HL = fastcluster.linkage(distH, method='average')
            dtH = 1.0
            indHL = sch.fcluster(HL, dtH*distH.max(), 'distance')
            freqHCl = Counter(indHL)
            # print ("hclust cut threshold:", dtH)
            # print ("n_clusters:", len(freqHCl))
            # print(freqHCl)
		
            npindHL = np.array(indHL)
            hcluster_score = {}
            for hclfreq in freqHCl.most_common(ntopics):
                hcl = hclfreq[0]
                hfreq = hclfreq[1]
                hcluster_score[hcl] = 0
                hclidx = (npindHL == hcl).nonzero()[0].tolist()
                for i in hclidx:
                    hcluster_score[hcl] += cluster_score[headline_to_cluster[headline_corpus[i]]]

            sorted_hclusters = sorted( ((v,k) for k,v in hcluster_score.items()), reverse=True)
            # print ("sorted hcluster_score:")
            # print (sorted_hclusters)

            for hscore, hcl in sorted_hclusters[:10]:
#                  print "\n(cluster, freq):", hcl, freqHCl[hcl]
                hclidx = (npindHL == hcl).nonzero()[0].tolist()
                final_cluster = []
                for i in hclidx:
                    final_cluster += tweet_cluster[i]
                final_clusters[count2] = final_cluster
                count2 = count2 + 1

                clean_headline = ''
                raw_headline = ''
                keywords = ''
                tids_set = set()
                tids_list = []
                urls_list = []
                selected_raw_tweets_set = set()
                tids_cluster = []
                for i in hclidx:
                    clean_headline += headline_corpus[i].replace(",", " ") + "//"
                    keywords += orig_headline_corpus[i].lower() + ","
                    tid = headline_to_tid[headline_corpus[i]]
                    tids_set.add(tid)
                    raw_tweet = tid_to_raw_tweet[tid].replace("\n", ' ').replace("\t", ' ')
                    raw_tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', raw_tweet)
                    selected_raw_tweets_set.add(raw_tweet.strip())
                    #fout.write("\nheadline tweet: " + raw_tweet.decode('utf8', 'ignore'))
                    tids_list.append(tid)
                    if tid_to_urls_window_corpus[tid]:
                        urls_list.append(tid_to_urls_window_corpus[tid])
                    for id in cluster_to_tids[headline_to_cluster[headline_corpus[i]]]:
                        tids_cluster.append(id)
                # print(tids_cluster) 				 	
                raw_headline = tid_to_raw_tweet[headline_to_tid[headline_corpus[hclidx[0]]]]
                raw_headline = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', raw_headline)
                raw_headline = raw_headline.replace("\n", ' ').replace("\t", ' ')
                keywords_list = str(sorted(list(set(keywords[:-1].split(",")))))[1:-1].replace('u\'','').replace('\'','')					

                #Select tweets with media urls
                #If need code to be more efficient, reduce the urls_list to size 1.	
                for tid in tids_cluster:
                    if len(urls_list) < 1 and tid_to_urls_window_corpus[tid] and tid not in tids_set:
                        raw_tweet = tid_to_raw_tweet[tid].replace("\n", ' ').replace("\t", ' ')
                        raw_tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', raw_tweet)
                        # raw_tweet = raw_tweet.decode('utf8', 'ignore')
                        #fout.write("\ncluster tweet: " + raw_tweet)
                        if raw_tweet.strip() not in selected_raw_tweets_set:
                            tids_list.append(tid)
                            urls_list.append(tid_to_urls_window_corpus[tid])
                            selected_raw_tweets_set.add(raw_tweet.strip())
                
                try:
                    print ("\n", clean_headline)
                except: pass

                urls_set = set()
                for url_list in urls_list:
                    for url in url_list:
                        urls_set.add(url)
                print("\n" + str(dtime) + "\t" + raw_headline + "\t" + keywords_list + "\t" + str(tids_list)[1:-1] + "\t" + str(list(urls_set))[1:-1][2:-1].replace('\'','').replace('uhttp','http'))
                # print(raw_headline)
            # print(final_clusters)
            window_corpus = []
            tids_window_corpus = []
            tid_to_urls_window_corpus = {}
            tid_to_raw_tweet = {}
            ntweets = 0
            # print("tfVocTimeWindows  ", tfVocTimeWindows)
            if t == 4:
                tfVocTimeWindows = {}
                t = 0
    file_timeordered_tweets.close()
