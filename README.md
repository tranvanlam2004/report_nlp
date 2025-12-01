# LAB 1 – TEXT TOKENIZATION

## 1. Mục tiêu
Bài lab đầu tiên nhằm cài đặt và thử nghiệm hai phương pháp tách từ cơ bản:
- **SimpleTokenizer**: tách theo khoảng trắng + xử lý nhẹ dấu câu.
- **RegexTokenizer**: tách bằng regex, linh hoạt và chính xác hơn.
- Thử nghiệm trên các câu mẫu và trên tập UD English EWT.

## 2. Nội dung thực hiện

### 2.1 SimpleTokenizer
- Chuyển văn bản về chữ thường.
- Tách theo khoảng trắng.
- Các dấu câu (`. , ! ?`) được tách thành token độc lập.

Ví dụ code:

class SimpleTokenizer:
    def tokenize(self, text):
        text = text.lower()
        for p in ".,?!":
            text = text.replace(p, f" {p} ")
        return [t for t in text.split() if t]
### 2.2 RegexTokenizer
Sử dụng regex để tách token theo quy luật:

Từ (chuỗi ký tự chữ/số)

Hoặc ký tự dấu câu tách riêng

Regex dùng:  \w+ | [^\w\s]
Code:

import re
class RegexTokenizer:
    pattern = re.compile(r"\w+|[^\w\s]")
    def tokenize(self, text):
        return self.pattern.findall(text.lower())
### 2.3 Kết quả thử nghiệm
Ví dụ câu:
Hello, world! This is a test.

Kết quả:
Simple: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Regex: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Với câu phức tạp hơn:
NLP is fascinating... isn't it?
Simple: ['nlp','is','fascinating','.','.','.',"isn't",'it','?']

Regex: ['nlp','is','fascinating','.','.','.','isn',"'",'t','it','?']

RegexTokenizer thể hiện khả năng “vỡ” các cấu trúc như từ có dấu apostrophe.

# LAB 2 – COUNT VECTORIZATION
## 1. Mục tiêu
Biểu diễn văn bản thành vector số theo mô hình Bag-of-Words:

Xây dựng CountVectorizer từ đầu.

Sử dụng tokenizer để tạo từ vựng.

Sinh ma trận document-term.

## 2. Cài đặt chính
### 2.1 Interface chung
class Vectorizer:
    def fit(self, docs): pass
    def transform(self, docs): pass
    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)
### 2.2 CountVectorizer
Duyệt qua toàn bộ corpus → thu thập vocabulary.
Gán chỉ số cho mỗi token.
Đếm số lần xuất hiện của token trong từng document.

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, docs):
        vocab = {}
        for doc in docs:
            for tok in self.tokenizer.tokenize(doc):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab = vocab

    def transform(self, docs):
        import numpy as np
        X = np.zeros((len(docs), len(self.vocab)), dtype=int)
        for i, doc in enumerate(docs):
            for tok in self.tokenizer.tokenize(doc):
                if tok in self.vocab:
                    X[i][self.vocab[tok]] += 1
        return X
###  2.3 Ví dụ chạy
Corpus:
I love NLP.
I love programming.
AI is a subfield of NLP.
Vocabulary mẫu:
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}
Document-term matrix:

[1,0,0,1,0,1,1,0,0,0]
[1,0,0,1,0,1,0,0,1,0]
[1,1,1,0,1,0,1,1,0,1]

# LAB 3 – WORD EMBEDDINGS
##1. Sử dụng mô hình pre-trained (GloVe)
Dùng lớp WordEmbedder để:

tải vector từ GloVe,

tính similarity,

truy vấn các từ gần nghĩa,

nhúng cả câu bằng trung bình vector.

Ví dụ:

vec = embedder.get_vector("king")
sim = embedder.get_similarity("king", "queen")
top = embedder.get_most_similar("computer", topn=10)

Kết quả similarity tiêu biểu:

Similarity(king, queen) = ~0.78
Similarity(king, man)   = ~0.53
Các từ gần “computer”:

computers, software, internet, digital, devices, applications...
2. Huấn luyện Word2Vec trên tập nhỏ
Pipeline:

Đọc file train UD EWT.

Tạo class SentenceStream để trả về danh sách tokens.

Huấn luyện Word2Vec:


Word2Vec(sentences, vector_size=100, window=5, min_count=3, sg=1)
Test từ gần nghĩa:

model.wv.most_similar("computer")
Do dữ liệu nhỏ → chất lượng embedding thấp → kết quả sai lệch.

3. Huấn luyện Word2Vec với Spark (tập lớn)
Pipeline Spark:

Raw text → Tokenizer → Word2Vec → Model
Cấu hình:

vectorSize = 100

minCount = 5

Kết quả:

similar("computer") → desktop, laptop, pc, tablet
Embedding mượt và hợp lý hơn nhờ dữ liệu lớn.

LAB 4 – GIẢM CHIỀU & TRỰC QUAN HÓA
1. PCA
Giữ cấu trúc toàn cục của embedding.

Các điểm tập trung nhiều, khó nhận ra cụm rõ ràng.

2. t-SNE
Nhấn mạnh quan hệ cục bộ.

Tạo các cụm từ rõ rệt: công nghệ, tên người, hành động,…

Quy trình:

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
PCA cho overview.

t-SNE cho insight về ngữ nghĩa nhóm từ.

LAB 5 – PHÂN LOẠI VĂN BẢN (TEXT CLASSIFICATION)
1. Thành phần đã cài đặt
1.1 Tokenizer + TF-IDF
TF-IDF được cài bằng công thức:

tfidf = tf * log((N+1)/(df+1)) + 1
Code giản lược:

idf = np.log((N+1)/(df+1)) + 1
tfidf = tf * idf
1.2 Logistic Regression Classifier

clf = LogisticRegression(solver="liblinear")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
2. CHẠY MÃ NGUỒN
2.1 Test cơ bản (Task 2)

python lab5_test.py
Kết quả:
Accuracy: 0.50
F1: 0.50
→ Do dataset quá nhỏ (16 mẫu).

2.2 Phân tích cảm xúc với PySpark (Task 3)

python lab5_spark_sentiment_analysis.py
Kết quả (5,791 mẫu):

Accuracy: 72.23%
F1: 0.7184
Confusion matrix:

TP = 563

TN = 238

FP = 178

FN = 130

→ Mô hình thiên lệch về Positive do imbalance.

2.3 Cải tiến mô hình (Task 4)

python lab5_improvement_test.py
Kết quả tóm tắt:

Mô hình	Accuracy	F1
Baseline (100 feat)	70.99%	0.7925
Naive Bayes	70.51%	0.7964
TF-IDF 200 features	73.00%	0.8060
Lọc từ hiếm	71.34%	0.7940

→ Best: TF-IDF 200 features.

3. PHÂN TÍCH CHUNG
Mô hình pre-trained (GloVe) cho kết quả tốt nhất về ngữ nghĩa.

Word2Vec Spark tốt hơn nhiều so với Word2Vec tập nhỏ.

PCA giúp xem tổng thể, t-SNE giúp hiểu cụm từ rõ hơn.

Logistic Regression + TF-IDF vẫn rất hiệu quả.

Tăng số lượng features giúp mô hình cải thiện rõ rệt.

4. KHÓ KHĂN
Spark hơi nặng, một số máy cài đặt mất thời gian.

Dataset nhỏ trong Task 2 làm kết quả kém.

Việc load GloVe bản text mất thời gian vì file lớn.

5. TÀI LIỆU THAM KHẢO
Tài liệu thư viện: Gensim, Scikit-Learn, PySpark MLlib

Gợi ý, hướng dẫn xây dựng code từ ChatGPT và DeepSeek

UD English EWT Dataset – Universal Dependencies

Stanford GloVe pretrained word vectors

#Lab6


# BÀI 1 – KHÔI PHỤC MASKED TOKEN (Masked Language Modeling)

## 1. Mô hình có dự đoán đúng từ “capital” không?

Có.  
Trong câu:  
Hanoi is the [MASK] of Vietnam.
BERT đã dự đoán đúng từ **“capital”**, và xác suất mô hình đưa ra rất cao (xấp xỉ 99%).  

Điều này cho thấy BERT hiểu mối quan hệ ngữ nghĩa giữa **Hanoi → capital → Vietnam**, và nắm rất tốt ngữ cảnh hai chiều trong câu.

## 2. Vì sao mô hình Encoder-only như BERT phù hợp cho tác vụ này?

Các lý do chính:

###  Nhìn ngữ cảnh hai chiều (Bidirectional Self-Attention)
BERT quan sát đồng thời cả phần **trước** và **sau** vị trí [MASK].  
Trong ví dụ:  
- Từ “Hanoi” gợi ý đó là một thành phố hoặc địa danh.  
- Cụm “of Vietnam” củng cố ý nghĩa rằng từ bị che phải là "capital".

###  Được huấn luyện đúng bài toán Masked LM
Trong giai đoạn pre-training, BERT thường xuyên phải dự đoán token bị ẩn đi.  
→ Kỹ năng này giúp mô hình cực kỳ mạnh trong việc suy đoán token dựa trên ngữ cảnh.

###  GPT không phù hợp cho nhiệm vụ MLM
GPT chỉ nhìn **một chiều** (trái → phải), không có quyền xem phần sau vị trí bị che.  
Vì vậy GPT **không thể** thực hiện Masked LM một cách tối ưu.

---

# BÀI 2 – DỰ ĐOÁN TỪ TIẾP THEO (Next Token Prediction)

## 1. Văn bản GPT sinh ra có hợp lý không?

Kết quả sinh văn bản của GPT thường:
- **trôi chảy**, ngữ pháp tốt, mạch văn tự nhiên;  
- Ý nghĩa nhìn chung hợp lý nhưng đôi khi **không chính xác về mặt thực tế**, vì GPT ưu tiên tính liền mạch hơn tính đúng ― đặc trưng của mô hình sinh văn bản.

## 2. Vì sao mô hình Decoder-only như GPT phù hợp với tác vụ sinh văn bản?

###  Cơ chế attention nhân quả (Causal Self-Attention)
GPT chỉ nhìn các token đã xuất hiện trước đó, đúng với cách con người viết từng từ một.  
Self-attention dạng nhân quả đảm bảo mô hình không dùng “thông tin tương lai”.

###  Mục tiêu huấn luyện trùng với bài toán Next Token Prediction
GPT được train để dự đoán **từ tiếp theo** trong chuỗi.  
→ Khi đưa một đoạn văn vào, mô hình tự nhiên tiếp tục nó theo đúng cách đã học.

###  Cơ chế sinh tự hồi quy (Autoregressive Generation)
GPT sinh token từng bước:  
p(x_t | x_1, x_2, ..., x_{t-1})

Điều này lý tưởng cho việc tạo câu, đoạn văn, hội thoại,…

###  Vì sao BERT không phù hợp?
- BERT không được huấn luyện để sinh từ tiếp theo.  
- Nếu dùng BERT để sinh chuỗi, mô hình phải dự đoán lại toàn bộ các token đã dùng, rất không tự nhiên.  
- BERT thiên về hiểu ngữ cảnh, không phải tạo văn bản.

---

# BÀI 3 – VECTOR BIỂU DIỄN CÂU (Sentence Embedding)

## 1. Kích thước vector và mối liên hệ với tham số của BERT

Khi đưa câu vào BERT, ta thu được:  
last_hidden_state.shape = [1, 8, 768]

Trong đó:
- 1 → batch size  
- 8 → số token sau khi tokenizer xử lý  
- **768** → kích thước vector (hidden_size của BERT-base)

→ Mỗi token được biểu diễn bằng một vector 768 chiều, phản ánh đặc trưng ngữ nghĩa trong không gian embedding của BERT.

## 2. Vì sao cần attention_mask khi Mean Pooling?

Tokenizer sẽ thêm padding cho những câu ngắn hơn.  
Padding token **không chứa thông tin**, nên nếu đem tính trung bình chung với token thật ⇒ vector câu sẽ bị “loãng” và sai lệch.

attention_mask cho phép:
- bỏ qua giá trị của padding khi tính mean;  
- đảm bảo vector cuối biểu diễn đúng nghĩa của câu thật.

Cách tính mean pooling hợp lệ:
sentence_vector = sum(hidden_states * attention_mask) / sum(attention_mask)
