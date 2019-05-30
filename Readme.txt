* Crawl tweet:
	python twitter_streaming.py > twitter_data.txt
* Convert tweet from json to text:
	python extract-json-to-text-stream.py twitter_data.txt clean_data.txt
* Process:
	python twitter_topics.py clean_data.txt 15


