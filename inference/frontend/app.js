const sourceText = document.getElementById("sourceText");
const translatedText = document.getElementById("translationResult");
const statusBox = document.getElementById("status");
const translateBtn = document.getElementById("translateBtn");
const incorrectBtn = document.getElementById("incorrectBtn");
const feedbackPanel = document.getElementById("feedbackPanel");
const correctedText = document.getElementById("correctedText");
const userId = document.getElementById("userId");
const saveFeedbackBtn = document.getElementById("saveFeedbackBtn");

let lastTranslation = "";

function setStatus(message) {
  statusBox.textContent = message;
}

setStatus("Ready.");

async function loadVersions() {
  try {
    const response = await fetch("/versions");
    if (response.ok) {
      const data = await response.json();
      const selectEl = document.getElementById("modelSelect");
      if (selectEl && data.versions) {
        selectEl.innerHTML = "";
        data.versions.forEach(v => {
          const opt = document.createElement("option");
          opt.value = v;
          opt.textContent = `Model Version: ${v.toUpperCase()}`;
          if (v === data.default) {
            opt.selected = true;
          }
          selectEl.appendChild(opt);
        });
      }
    }
  } catch (error) {
    console.error("Failed to load model versions:", error);
  }
}

// Load versions on page load
loadVersions();

translateBtn.addEventListener("click", async () => {
  const text = sourceText.value.trim();
  if (!text) {
    setStatus("Add Akkadian text before translating.");
    return;
  }

  setStatus("Translating...");
  translatedText.textContent = "Working...";
  incorrectBtn.disabled = true;
  feedbackPanel.classList.add("hidden");

  try {
    const modelSelectEl = document.getElementById("modelSelect");
    const model_version = modelSelectEl ? modelSelectEl.value : "v2";
    const response = await fetch("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model_version }),
    });

    if (!response.ok) {
      let errDetail = `Translation failed with status ${response.status}`;
      try {
        const errData = await response.json();
        if (errData && errData.detail) {
          errDetail = errData.detail;
        }
      } catch (e) {}
      throw new Error(errDetail);
    }

    const data = await response.json();
    lastTranslation = data.translation || "";
    translatedText.textContent = lastTranslation || "No translation returned.";
    incorrectBtn.disabled = false;
    setStatus(`Translated with ${data.model_source} (Version: ${data.model_version || model_version}).`);
  } catch (error) {
    translatedText.textContent = "Translation failed.";
    setStatus(error.message);
  }
});

incorrectBtn.addEventListener("click", () => {
  feedbackPanel.classList.toggle("hidden");
  if (!feedbackPanel.classList.contains("hidden")) {
    correctedText.focus();
  }
});

saveFeedbackBtn.addEventListener("click", async () => {
  const source = sourceText.value.trim();
  const correction = correctedText.value.trim();

  if (!source || !correction) {
    setStatus("Add both the source text and the corrected translation.");
    return;
  }

  setStatus("Saving feedback...");

  try {
    const response = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_text: source,
        corrected_text: correction,
        translated_text: lastTranslation,
        user_id: userId.value.trim() || null,
      }),
    });

    if (!response.ok) {
      throw new Error(`Feedback save failed with status ${response.status}`);
    }

    const data = await response.json();
    setStatus(`Feedback saved as row ${data.id}.`);
    feedbackPanel.classList.add("hidden");
    correctedText.value = "";
  } catch (error) {
    setStatus(error.message);
  }
});

// Quick examples handler
document.querySelectorAll(".example-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    sourceText.value = chip.getAttribute("data-text");
    translateBtn.click();
  });
});