* Lấy dữ liệu:
	python twitter_streaming.py > twitter_data.txt
* Lấy những trường cần xử lý trong Tweet.json:
	python extract-json-to-text-stream.py twitter_data.txt clean_data.txt

* Xử lý:
	python twitter_topics.py clean_data.txt 15


* Phân cụm bằng hierachical(phân tầng) -> dendrogram(cây phân tầng)
	- Cắt bằng cách lấy threshold = 0.5(cao hơn có thể kết quả k tốt, nhiều  topic khác nhau ở trong cùng 1 cụm - thấp hơn thì kết quả có thể tốt hơn nhưng topic bị phân mảnh hơn)

*Xếp hạng kết quả sau khi phân cụm:
	- Tính điểm và xếp hạng đối với mỗi cửa sổ thời gian(time_window: 15') để lấy ra top topic trong khung thời gian đó
	- Do topic này có thể xuất hiên ở các cửa sổ thời gian khác -  tính cả trong time_window trước: lấy 4 time_window trước(với trọng số nhỏ hơn so với time_window hiện tại)

