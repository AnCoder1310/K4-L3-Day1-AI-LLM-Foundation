# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 → 1.5, câu trả lời thường trở nên đa dạng, sáng tạo và ít đoán trước hơn. Ở temperature thấp, mô hình có xu hướng trả lời ổn định và nhất quán; ở temperature cao, cách diễn đạt và nội dung có thể thay đổi nhiều hơn, đôi khi cũng dễ xuất hiện thông tin kém chính xác hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ chọn temperature khoảng 0.2–0.3 cho chatbot hỗ trợ khách hàng vì cần câu trả lời ổn định, chính xác và nhất quán, hạn chế việc AI trả lời quá sáng tạo hoặc thay đổi nội dung giữa các lần hỏi giống nhau.
### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o có thể đắt hơn GPT-4o-mini khoảng 10–20 lần tùy loại token và mức giá API tại thời điểm tính. Với workload 10,5 triệu token đầu ra/ngày, nên dùng GPT-4o khi cần chất lượng suy luận cao và xử lý yêu cầu phức tạp; còn GPT-4o-mini phù hợp cho chatbot CSKH thông thường, hỏi đáp đơn giản để tiết kiệm chi phí.


---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Khi là giáo viên tiểu học, em sẽ giải thích ngắn gọn, dễ hiểu, sử dụng từ vựng đơn giản và các ví dụ gần gũi với trẻ 8 tuổi. Ngược lại, nếu là chuyên gia tài chính, em sẽ giải thích dài và sâu hơn, sử dụng nhiều thuật ngữ kỹ thuật và ví dụ liên quan đến tài chính. System prompt định hướng cách model lựa chọn từ ngữ, mức độ chi tiết, giọng điệu và cách trình bày câu trả lời. Vì vậy, cùng một câu hỏi nhưng với các persona khác nhau, model có thể tạo ra những phản hồi rất khác nhau.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với một đoạn văn tiếng Việt khoảng 100 từ, tiktoken có thể cho khoảng 130–150 token, trong khi cách ước lượng số từ / 0.75 cho khoảng 133 token. Hai con số có thể chênh nhau khoảng 5–10%, tùy vào nội dung cụ thể của đoạn văn. Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì cách mã hóa của tokenizer có thể tách các từ và ký tự tiếng Việt thành nhiều token hơn, đặc biệt với các ký tự có dấu. Vì vậy, không nên dùng số từ để ước lượng token một cách tuyệt đối khi tính chi phí API.


---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng khi người dùng cần nhận kết quả nhanh và nội dung được sinh ra tương đối dài, vì họ có thể đọc từng phần ngay khi model đang xử lý thay vì phải chờ toàn bộ câu trả lời. Điều này giúp giảm cảm giác chờ đợi và cải thiện trải nghiệm tương tác, đặc biệt với chatbot. Ngược lại, non-streaming phù hợp với các tác vụ cần nhận toàn bộ kết quả một lần, chẳng hạn như xử lý dữ liệu, lưu kết quả vào hệ thống hoặc khi ứng dụng không cần hiển thị nội dung theo thời gian thực.


### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp client tăng dần thời gian chờ giữa các lần retry, từ đó giảm áp lực lên API khi hệ thống đang quá tải và tạo cơ hội để hệ thống phục hồi. Nếu hàng nghìn client cùng retry với delay cố định 1 giây, chúng có thể gửi request lại gần như cùng lúc, tạo ra một đợt tải lớn mới và khiến tình trạng quá tải kéo dài. Vì vậy, exponential backoff thường kết hợp thêm jitter (độ trễ ngẫu nhiên) để tránh hiện tượng nhiều client retry đồng thời.


---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Em chọn persona là trợ lý học tập AI, có nhiệm vụ hỗ trợ sinh viên giải thích kiến thức và hướng dẫn học tập.
System prompt: "Bạn là một trợ lý học tập AI thân thiện. Hãy giải thích kiến thức rõ ràng, dễ hiểu, ưu tiên ví dụ thực tế và trả lời bằng tiếng Việt. Với câu hỏi đơn giản, trả lời ngắn gọn; với câu hỏi phức tạp, trình bày theo từng bước."

Em yêu cầu "trả lời bằng tiếng Việt" để câu trả lời phù hợp với người dùng Việt Nam. Cụm "giải thích rõ ràng, dễ hiểu" giúp model hạn chế sử dụng thuật ngữ quá phức tạp và phù hợp hơn với mục đích học tập.


### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là history chỉ lưu được 3 lượt hội thoại gần nhất, nên model có thể quên những thông tin quan trọng được đề cập từ trước. Em đề xuất xây dựng bộ nhớ dài hạn bằng cách lưu các thông tin quan trọng của người dùng vào cơ sở dữ liệu hoặc vector database. Khi người dùng đặt câu hỏi mới, hệ thống sẽ tìm kiếm những thông tin liên quan trong bộ nhớ và đưa chúng vào prompt để model có thể trả lời phù hợp với ngữ cảnh trước đó.


---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
