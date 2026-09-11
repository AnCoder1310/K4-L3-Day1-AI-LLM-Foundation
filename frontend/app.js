const form = document.querySelector('.question-box');
const questionInput = document.querySelector('#question');
const apiKeyInput = document.querySelector('#api-key');
const submitButton = form.querySelector('button[type="button"]');
const status = document.querySelector('#answer-status');
const questionPreview = document.querySelector('#question-preview');
const flashAnswer = document.querySelector('#flash-answer');
const liteAnswer = document.querySelector('#lite-answer');
const flashLatency = document.querySelector('#flash-latency');
const liteLatency = document.querySelector('#lite-latency');

const models = [
  { id: 'gemini-2.5-flash', answer: flashAnswer, latency: flashLatency },
  { id: 'gemini-2.5-flash-lite', answer: liteAnswer, latency: liteLatency },
];

function readResponse(data) {
  return data.candidates?.[0]?.content?.parts?.map((part) => part.text || '').join('') || 'Model không trả về nội dung.';
}

async function askModel(model, question, apiKey) {
  const startedAt = performance.now();
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents: [{ parts: [{ text: question }] }] }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Gemini trả về lỗi ${response.status}.`);
  }

  return {
    text: readResponse(await response.json()),
    latency: `${((performance.now() - startedAt) / 1000).toFixed(2)}s`,
  };
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
});

submitButton.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  const apiKey = apiKeyInput.value.trim();

  if (!question) {
    status.textContent = 'Hãy nhập câu hỏi trước';
    questionInput.focus();
    return;
  }

  if (!apiKey) {
    status.textContent = 'Hãy nhập Gemini API key';
    apiKeyInput.focus();
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = 'Đang hỏi Gemini...';
  status.textContent = 'Gemini đang xử lý câu hỏi';
  questionPreview.textContent = question;
  flashAnswer.textContent = 'Đang chờ Gemini Flash...';
  liteAnswer.textContent = 'Đang chờ Gemini Flash Lite...';

  try {
    const results = await Promise.all(models.map((model) => askModel(model.id, question, apiKey)));
    results.forEach((result, index) => {
      models[index].answer.textContent = result.text;
      models[index].latency.textContent = result.latency;
    });
    status.textContent = 'Đã nhận câu trả lời';
    document.querySelector('#answers').scrollIntoView({ behavior: 'smooth' });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Không thể kết nối Gemini.';
    status.textContent = message;
    flashAnswer.textContent = 'Chưa thể nhận câu trả lời.';
    liteAnswer.textContent = 'Chưa thể nhận câu trả lời.';
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = 'Gửi câu hỏi <span aria-hidden="true">→</span>';
  }
});
