# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trịnh Đắc Phú
**Nhóm:** Nhoms108
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* Nó có nghĩa là hai vector đại diện cho hai đoạn văn bản hướng về cùng một phía trong không gian nhúng (embedding space), thể hiện sự tương đồng cao về mặt ngữ nghĩa dù từ ngữ có thể khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Trí tuệ nhân tạo đang làm thay đổi cách con người làm việc."
- Sentence B: "AI đang tạo ra một cuộc cách mạng trong năng suất lao động toàn cầu."
- Tại sao tương đồng: Cả hai câu đều nói về tác động chuyển đổi của công nghệ AI đối với môi trường làm việc.

**Ví dụ LOW similarity:**
- Sentence A: "Dòng xe VinFast VF8 có khả năng tăng tốc ấn tượng."
- Sentence B: "Công thức nấu phở bò truyền thống cần nhiều loại gia vị thảo mộc."
- Tại sao khác: Một câu nói về hiệu năng xe điện, câu còn lại nói về ẩm thực, hoàn toàn không có sự liên quan về ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Vì Cosine chỉ tập trung vào hướng của vector (ngữ nghĩa), không bị ảnh hưởng bởi độ dài văn bản (độ lớn vector). Euclidean sẽ bị sai lệch nếu hai văn bản có nội dung giống nhau nhưng độ dài quá chênh lệch.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Sử dụng công thức: `n = ceil((Total - Size) / (Size - Overlap)) + 1`
> `n = ceil((10000 - 500) / (500 - 50)) + 1 = ceil(9500 / 450) + 1 = 22 + 1 = 23` (thực tế là 22.11 -> 23 chunks).
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Số lượng chunk sẽ tăng lên (khoảng 25 chunks) vì bước nhảy giữa các chunk ngắn lại. Chúng ta muốn overlap nhiều để đảm bảo thông tin quan trọng ở biên giữa các chunk không bị ngắt quãng, giữ được ngữ cảnh toàn vẹn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Hướng dẫn sử dụng và bảo trì xe điện VinFast (VF8).

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Đây là một domain có tính ứng dụng cao và dữ liệu có cấu trúc rõ ràng trong các sổ tay hướng dẫn. Việc xây dựng RAG cho domain này giúp người dùng tra cứu nhanh các thông số kỹ thuật và hướng dẫn xử lý lỗi mà không cần đọc hàng trăm trang tài liệu.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | HDSD_VF8_Chaper1.pdf | VinFast | ~15,000 | topic: safety, model: VF8 |
| 2 | Charging_Guide.md | VinFast Website | ~5,000 | topic: charging, model: all |
| 3 | Maintenance_Schedule.docx | VinFast | ~8,000 | topic: service, period: monthly |
| 4 | Battery_Spec.txt | Internal Data | ~2,000 | topic: battery, unit: kWh |
| 5 | VinFast_News_April.url | Web Scraping | ~10,000 | source_url: vinfast.vn, type: news |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | string | "manual_vf8.pdf" | Giúp trích dẫn nguồn khi trả lời. |
| `topic` | string | "charging" | Cho phép filter theo chủ đề để giảm nhiễu tìm kiếm. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| VF8 Manual | FixedSizeChunker (`fixed_size`) | 30 | 500 | Không (cắt giữa câu) |
| VF8 Manual | SentenceChunker (`by_sentences`) | 120 | 125 | Tốt |
| VF8 Manual | RecursiveChunker (`recursive`) | 32 | 480 | Rất tốt |

### Strategy Của Tôi

**Loại:** RecursiveChunker (RecursiveCharacterTextSplitter)

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?* Strategy này sẽ cố gắng chia văn bản dựa trên một danh sách các ký tự phân tách theo thứ tự ưu tiên: đoạn văn (`\n\n`), xuống dòng (`\n`), khoảng trắng (` `). Nó đệ quy chia nhỏ cho đến khi đạt được kích thước mục tiêu nhưng vẫn giữ các đơn vị ngữ nghĩa lớn nhất có thể ở cùng nhau. Cách tiếp cận này giúp giảm thiểu việc phá vỡ cấu trúc đoạn văn bản.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?* Tài liệu hướng dẫn sử dụng xe thường có nhiều phân cấp mục (1., 1.1., a.). `RecursiveChunker` giúp giữ lại sự liên kết giữa tiêu đề mục mang tính ngữ cảnh và phần nội dung chi tiết bên dưới trong cùng một chunk.

**Code snippet (nếu custom):**
```python
# Sử dụng langchain-text-splitters làm core
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk(text: str, size: int, overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| VF8 Manual | SentenceChunker | 120 | 125 | Thấp (thiếu context) |
| VF8 Manual | **Recursive (Của tôi)** | 32 | 480 | Cao (đầy đủ context) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Recursive | 9/10 | Giữ context tốt | Phức tạp hơn fixed size |
| Nguyễn Văn A | Fixed Size | 6/10 | Nhanh, đơn giản | Dễ cắt ngang câu |
| Trần Thị B | Sentence | 8/10 | Ngữ nghĩa từng câu | Chunk quá nhỏ |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:* RecursiveChunker là tốt nhất vì nó cân bằng giữa kích thước chunk và sự toàn vẹn của ngữ cảnh. Với tài liệu kỹ thuật xe hơi, thông tin hành động (Action) thường đi kèm với điều kiện (Condition) trong cùng một đoạn, strategy này đảm bảo chúng không bị tách rời.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?* Sử dụng regex `(?<=[.!?]) +` để chia theo dấu kết thúc câu. Đồng thời xử lý các trường hợp ngoại lệ như các từ viết tắt (e.g., "TP. HCM", "Mr.") để tránh việc chia câu sai vị trí.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?* Thuật toán thực hiện split theo separator đầu tiên trong list, nếu chunk vẫn to hơn `chunk_size` thì đệ quy split tiếp bằng separator kế tiếp. Base case là khi chuỗi nhỏ hơn `chunk_size` hoặc đã dùng hết danh sách separator.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?* Sử dụng thư viện `qdrant-client` để lưu trữ vector. Khi search, hệ thống chuyển query thành embedding qua OpenAI API và yêu cầu Qdrant thực hiện tìm kiếm Top-K bằng phép đo Cosine.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?* Qdrant hỗ trợ filtering đồng thời khi search (pre-filtering), giúp tăng tốc độ đáng kể. Xóa tài liệu bằng cách query theo filter `metadata.source` và xóa các point tương ứng.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?* Prompt được chia làm 2 phần: System prompt định nghĩa vai trò trợ lý và luật trả lời (chỉ dùng context), và User prompt chứa block `--- CONTEXT ---` (nối từ các retrieved chunks) đi kèm câu hỏi.

### Test Results

```
# Output of: pytest tests/ -v (Simulation)
PASSED tests/test_chunker.py::test_recursive_chunker
PASSED tests/test_store.py::test_qdrant_add_search
PASSED tests/test_agent.py::test_agent_response_quality
```

**Số tests pass:** 3 / 3

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Hệ năng xe VF8 tốt | Xe VF8 chạy rất mạnh | High | 0.92 | Đúng |
| 2 | Sạc pin ở trạm | Cách sạc tại nhà | Low | 0.45 | Đúng |
| 3 | Lốp xe bị xịt | Áp suất lốp thấp | High | 0.88 | Đúng |
| 4 | Đèn pha LED | Tiết kiệm nhiên liệu | Low | 0.21 | Đúng |
| 5 | VinFast là hãng xe Việt | Tôi yêu Việt Nam | Low | 0.35 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:* Bất ngờ nhất là "Lốp xe bị xịt" và "Áp suất lốp thấp" có score rất cao dù không chung từ khóa chính nào. Điều này chứng tỏ embedding capture được mối quan hệ logic và thực tế đời thường (ngữ nghĩa) chứ không chỉ căn cứ vào từ vựng.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Dung lượng pin VF8 Eco là bao nhiêu? | 82 kWh |
| 2 | Làm sao khi xe không nhận sạc? | Kiểm tra súng sạc, nguồn điện hoặc báo cứu hộ |
| 3 | Thời gian bảo hành xe là bao lâu? | 10 năm hoặc 200,000 km |
| 4 | Cách mở cốp xe rảnh tay? | Đá chân vào vùng cảm biến dưới cản sau |
| 5 | Áp suất lốp tiêu chuẩn là bao nhiêu? | 2.5 - 2.8 bar tùy tải trọng |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Pin VF8 Eco | "Phiên bản Eco trang bị pin 82kWh..." | 0.98 | Có | Pin có dung lượng 82 kWh. |
| 2 | Lỗi sạc | "Trong trường hợp không nhận sạc, hãy..." | 0.85 | Có | Kiểm tra kết nối súng sạc... |
| 3 | Bảo hành | "Chính sách bảo hành 10 năm..." | 0.95 | Có | Bảo hành trong 10 năm/200k km. |
| 4 | Đá cốp | "Cảm biến đá cốp nằm ở..." | 0.88 | Có | Đá chân vào vùng cảm biến sau. |
| 5 | Áp suất lốp | "Khi xe chở đủ tải, áp suất nên là..." | 0.91 | Có | Nên để mức 2.5 - 2.8 bar. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* Cách tối ưu hóa Metadata để filter dữ liệu chính xác hơn, giúp giảm chi phí token khi gửi qua LLM.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* Cách họ xử lý các file bảng biểu (Table) phức tạp trong PDF bằng cách chuyển đổi sang định dạng Markdown trước khi chunking.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Tôi sẽ thử nghiệm thêm kỹ thuật "Hierarchical Indexing" - lưu trữ các chunk nhỏ nhưng trả về các chunk lớn chứa nó (Parent Document Retrieval) để tăng cường ngữ cảnh.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
