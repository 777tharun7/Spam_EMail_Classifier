const SAMPLES = [
  {
    type: "spam",
    subject: "Winner! Your email was selected",
    body: "Congratulations! You have won a free cruise trip. Claim your prize now before it expires!",
  },
  {
    type: "spam",
    subject: "Alert: Unusual login detected",
    body: "Urgent! Unusual login detected. Verify your account immediately to secure your account.",
  },
  {
    type: "spam",
    subject: "Limited slots for free crypto training",
    body: "Act fast! Earn cash daily with zero experience. Start today!",
  },
  {
    type: "ham",
    subject: "Team dinner tonight",
    body: "Hey, don't forget our dinner plan at 8. Will you be joining the team dinner tonight?",
  },
  {
    type: "ham",
    subject: "Project files uploaded",
    body: "I uploaded the project files to the shared folder. Let's review the code changes after lunch.",
  },
  {
    type: "ham",
    subject: "Call reschedule",
    body: "Can we reschedule the call to tomorrow afternoon? Meeting went well, I'll send summary notes.",
  },
];

let modelData = null;
let stopWordsSet = new Set();
const scanHistory = [];

const form = document.getElementById("classifier-form");
const subjectInput = document.getElementById("subject-input");
const messageInput = document.getElementById("message-input");
const charCount = document.getElementById("char-count");
const submitBtn = document.getElementById("submit-btn");
const clearBtn = document.getElementById("clear-btn");
const errorMsg = document.getElementById("error-msg");
const emptyState = document.getElementById("empty-state");
const resultContent = document.getElementById("result-content");
const verdictBanner = document.getElementById("verdict-banner");
const verdictTitle = document.getElementById("verdict-title");
const confidenceBadge = document.getElementById("confidence-badge");
const spamPct = document.getElementById("spam-pct");
const hamPct = document.getElementById("ham-pct");
const spamBar = document.getElementById("spam-bar");
const hamBar = document.getElementById("ham-bar");
const tokenList = document.getElementById("token-list");
const cleanedTextEl = document.getElementById("cleaned-text");
const examplesContainer = document.getElementById("examples-container");
const historyList = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history-btn");

function cleanText(text) {
  const lowered = text.toLowerCase();
  const noDigits = lowered.replace(/\d+/g, "");
  // Match Python string.punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
  const noPunct = noDigits.replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, "");
  return noPunct.trim().replace(/\s+/g, " ");
}

function tokenize(cleaned) {
  // Match scikit-learn default token_pattern: r"(?u)\b\w\w+\b"
  const matches = cleaned.match(/[a-zA-Z0-9_]{2,}/g) || [];
  return matches.filter((tok) => !stopWordsSet.has(tok));
}

function classifyWithModel(rawText) {
  const cleaned = cleanText(rawText);
  if (!cleaned) {
    return null;
  }

  const tokens = tokenize(cleaned);
  const tf = new Map();
  for (const tok of tokens) {
    if (Object.prototype.hasOwnProperty.call(modelData.tokens, tok)) {
      tf.set(tok, (tf.get(tok) || 0) + 1);
    }
  }

  // Compute TF-IDF vector with L2 normalization matching TfidfVectorizer(norm='l2')
  let sqSum = 0;
  const tfidf = new Map();
  for (const [tok, count] of tf.entries()) {
    const weight = count * modelData.tokens[tok].idf;
    tfidf.set(tok, weight);
    sqSum += weight * weight;
  }

  const norm = sqSum > 0 ? Math.sqrt(sqSum) : 1;
  let logHam = modelData.class_log_prior[0];
  let logSpam = modelData.class_log_prior[1];

  const signals = [];
  for (const [tok, rawWeight] of tfidf.entries()) {
    const x = rawWeight / norm;
    const info = modelData.tokens[tok];
    logHam += x * info.ham;
    logSpam += x * info.spam;
    signals.push({
      token: tok,
      logOdds: info.spam - info.ham,
      lean: info.spam >= info.ham ? "spam" : "ham",
    });
  }

  // Log-sum-exp for exact posterior probabilities
  const maxLog = Math.max(logHam, logSpam);
  const expHam = Math.exp(logHam - maxLog);
  const expSpam = Math.exp(logSpam - maxLog);
  const sumExp = expHam + expSpam;

  const hamProb = expHam / sumExp;
  const spamProb = expSpam / sumExp;

  signals.sort((a, b) => Math.abs(b.logOdds) - Math.abs(a.logOdds));

  return {
    isSpam: spamProb >= 0.5,
    spamProb,
    hamProb,
    cleaned,
    signals,
  };
}

function updateCounts() {
  const text = messageInput.value;
  const trimmed = text.trim();
  const words = trimmed ? trimmed.split(/\s+/).length : 0;
  charCount.textContent = `${words} word${words === 1 ? "" : "s"} · ${text.length} chars`;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add("visible");
}

function clearError() {
  errorMsg.textContent = "";
  errorMsg.classList.remove("visible");
}

function renderResult(result, combinedText) {
  emptyState.classList.add("hidden");
  resultContent.classList.remove("hidden");

  const spamPercent = Math.round(result.spamProb * 1000) / 10;
  const hamPercent = Math.round(result.hamProb * 1000) / 10;
  const topPercent = result.isSpam ? spamPercent : hamPercent;

  verdictBanner.className = `verdict-banner ${result.isSpam ? "spam" : "ham"}`;
  verdictTitle.textContent = result.isSpam ? "Spam Detected" : "Legitimate (Ham)";
  confidenceBadge.textContent = `${topPercent}% confidence`;

  spamPct.textContent = `${spamPercent}%`;
  hamPct.textContent = `${hamPercent}%`;
  spamBar.style.width = `${spamPercent}%`;
  hamBar.style.width = `${hamPercent}%`;

  tokenList.textContent = "";
  if (result.signals.length === 0) {
    const span = document.createElement("span");
    span.className = "token-empty";
    span.textContent = "No words matched the 94-token training vocabulary (using prior probabilities).";
    tokenList.appendChild(span);
  } else {
    for (const sig of result.signals) {
      const pill = document.createElement("span");
      pill.className = `token-pill ${sig.lean}`;
      pill.textContent = `${sig.token} (${sig.lean === "spam" ? "+spam" : "+ham"})`;
      tokenList.appendChild(pill);
    }
  }

  cleanedTextEl.textContent = result.cleaned;
  addHistoryItem(combinedText, result.isSpam, topPercent);
}

function addHistoryItem(text, isSpam, confidence) {
  const subject = subjectInput.value.trim();
  const body = messageInput.value.trim();
  if (scanHistory.length > 0 && scanHistory[0].combined === text) {
    return;
  }

  scanHistory.unshift({
    subject,
    body,
    combined: text,
    isSpam,
    confidence,
  });
  if (scanHistory.length > 6) {
    scanHistory.pop();
  }
  renderHistory();
}

function renderHistory() {
  historyList.textContent = "";
  if (scanHistory.length === 0) {
    clearHistoryBtn.classList.add("hidden");
    const p = document.createElement("p");
    p.className = "history-empty";
    p.textContent = "No scans yet in this session.";
    historyList.appendChild(p);
    return;
  }

  clearHistoryBtn.classList.remove("hidden");
  for (const item of scanHistory) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "history-item";

    const textSpan = document.createElement("span");
    textSpan.className = "history-text";
    textSpan.textContent = item.subject || item.body;

    const tagSpan = document.createElement("span");
    tagSpan.className = `history-tag ${item.isSpam ? "spam" : "ham"}`;
    tagSpan.textContent = `${item.isSpam ? "Spam" : "Ham"} · ${item.confidence}%`;

    btn.appendChild(textSpan);
    btn.appendChild(tagSpan);
    btn.addEventListener("click", () => {
      subjectInput.value = item.subject;
      messageInput.value = item.body;
      updateCounts();
      runAnalysis();
    });
    historyList.appendChild(btn);
  }
}

function runAnalysis() {
  clearError();
  if (!modelData) {
    showError("Model is still loading. Please try again in a moment.");
    return;
  }

  const combined = [subjectInput.value.trim(), messageInput.value.trim()]
    .filter(Boolean)
    .join(" ");

  if (!combined) {
    showError("Please enter an email subject or message body to analyze.");
    return;
  }

  const result = classifyWithModel(combined);
  if (!result) {
    showError("Message contains no classifiable words after text normalization.");
    return;
  }

  renderResult(result, combined);
}

function loadSample(sample) {
  subjectInput.value = sample.subject;
  messageInput.value = sample.body;
  updateCounts();
  runAnalysis();
}

function initExamples() {
  for (const sample of SAMPLES) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = `${sample.type === "spam" ? "Spam" : "Ham"}: ${sample.subject}`;
    btn.addEventListener("click", () => loadSample(sample));
    examplesContainer.appendChild(btn);
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  runAnalysis();
});

messageInput.addEventListener("input", updateCounts);

clearBtn.addEventListener("click", () => {
  subjectInput.value = "";
  messageInput.value = "";
  updateCounts();
  clearError();
  resultContent.classList.add("hidden");
  emptyState.classList.remove("hidden");
  messageInput.focus();
});

document.getElementById("btn-sample-spam").addEventListener("click", () => {
  loadSample(SAMPLES[0]);
});

document.getElementById("btn-sample-ham").addEventListener("click", () => {
  loadSample(SAMPLES[3]);
});

clearHistoryBtn.addEventListener("click", () => {
  scanHistory.length = 0;
  renderHistory();
});

async function bootstrap() {
  initExamples();
  updateCounts();
  submitBtn.disabled = true;
  submitBtn.textContent = "Loading Model...";

  try {
    const res = await fetch("./model.json");
    if (!res.ok) {
      throw new Error("Failed to load model.json");
    }
    modelData = await res.json();
    stopWordsSet = new Set(modelData.stop_words || []);
    submitBtn.disabled = false;
    submitBtn.textContent = "Analyze Email";
  } catch (err) {
    showError(err.message || "Could not load classifier model.");
    submitBtn.textContent = "Analyze Email";
  }
}

bootstrap();
