const uploadForm = document.getElementById('uploadForm');
const card = document.querySelector('.card');
const laxInput = document.getElementById('lax');
const flexedInput = document.getElementById('flexed');
const changeImagesButton = document.getElementById('changeImagesButton');
const loadingState = document.getElementById('loadingState');
const errorState = document.getElementById('errorState');
const resultContainer = document.getElementById('resultContainer');
const resultHeading = document.getElementById('resultHeading');
const resultDetails = document.getElementById('resultDetails');
const resultImage = document.getElementById('resultImage');
const altText = document.getElementById('altText');
const thumbUpButton = document.getElementById('thumbUpButton');
const thumbDownButton = document.getElementById('thumbDownButton');
const correctionPanel = document.getElementById('correctionPanel');
const correctionText = document.getElementById('correctionText');
const submitCorrection = document.getElementById('submitCorrection');
const feedbackStatus = document.getElementById('feedbackStatus');

let currentResult = null;
let currentLaxFile = null;
let currentFlexedFile = null;

async function readResponse(response) {
    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json')
        ? await response.json()
        : { detail: await response.text() };

    if (!response.ok) {
        throw new Error(payload.detail || `Request failed with HTTP ${response.status}.`);
    }
    return payload;
}

function setFeedbackStatus(message, isError = false) {
    feedbackStatus.textContent = message;
    feedbackStatus.classList.toggle('is-error', isError);
}

function setFeedbackSelection(selection = null) {
    const upSelected = selection === 'up';
    const downSelected = selection === 'down';
    thumbUpButton.classList.toggle('is-selected', upSelected);
    thumbDownButton.classList.toggle('is-selected', downSelected);
    thumbUpButton.setAttribute('aria-pressed', String(upSelected));
    thumbDownButton.setAttribute('aria-pressed', String(downSelected));
}

function resetFeedback(message = '') {
    setFeedbackSelection();
    correctionPanel.hidden = true;
    correctionText.value = '';
    setFeedbackStatus(message);
}

function replayResultTextAnimation() {
    const animatedItems = [
        resultHeading,
        ...resultDetails.querySelectorAll('.meaning-panel, .feedback-panel'),
    ];
    animatedItems.forEach((item) => item.classList.remove('is-entering'));
    void resultDetails.offsetWidth;
    animatedItems.forEach((item) => item.classList.add('is-entering'));
}

function clearResult() {
    card.classList.remove('has-result');
    resultContainer.style.display = 'none';
    resultHeading.hidden = true;
    resultDetails.hidden = true;
    resultImage.classList.remove('is-visible');
    resultImage.removeAttribute('src');
    resultImage.alt = 'EMS Output';
    altText.textContent = '';
    currentResult = null;
    currentLaxFile = null;
    currentFlexedFile = null;
    laxInput.value = '';
    flexedInput.value = '';
    loadingState.style.display = 'none';
    errorState.style.display = 'none';
    errorState.textContent = '';
    uploadForm.hidden = false;
    changeImagesButton.hidden = true;
    resetFeedback();
}

async function displayResult(result) {
    currentResult = result;
    resultImage.classList.remove('is-visible');
    resultImage.alt = result.alt_text;
    resultImage.src = `data:image/jpeg;base64,${result.image_base64}`;
    if (typeof resultImage.decode === 'function') {
        await resultImage.decode().catch(() => {});
    }
    altText.textContent = result.alt_text;
    card.classList.add('has-result');
    resultContainer.style.display = 'block';
    resultHeading.hidden = false;
    resultDetails.hidden = false;
    uploadForm.hidden = true;
    changeImagesButton.hidden = false;
    replayResultTextAnimation();
    requestAnimationFrame(() => {
        requestAnimationFrame(() => resultImage.classList.add('is-visible'));
    });
}

async function recordFeedback(accurate) {
    if (!currentResult) {
        return;
    }
    const response = await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            analysis_id: currentResult.analysis_id,
            accurate,
        }),
    });
    await readResponse(response);
}

uploadForm.onsubmit = async (event) => {
    event.preventDefault();

    currentLaxFile = laxInput.files[0];
    currentFlexedFile = flexedInput.files[0];

    const formData = new FormData();
    formData.append('lax_image', currentLaxFile);
    formData.append('flexed_image', currentFlexedFile);

    loadingState.style.display = 'block';
    resultContainer.style.display = 'none';
    resultHeading.hidden = true;
    resultDetails.hidden = true;
    changeImagesButton.hidden = true;
    errorState.style.display = 'none';
    currentResult = null;
    resetFeedback();

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData,
        });
        const result = await readResponse(response);
        await displayResult(result);
    } catch (error) {
        console.error('Error:', error);
        errorState.textContent = error.message
            || 'Connection lost. Check that the API server is running.';
        errorState.style.display = 'block';
    } finally {
        loadingState.style.display = 'none';
    }
};

changeImagesButton.addEventListener('click', () => {
    clearResult();
    laxInput.focus();
});

thumbUpButton.addEventListener('click', async () => {
    setFeedbackSelection('up');
    correctionPanel.hidden = true;
    setFeedbackStatus('Sending feedback...');

    try {
        await recordFeedback(true);
        setFeedbackStatus('Thanks - feedback recorded.');
    } catch (error) {
        console.error('Feedback error:', error);
        setFeedbackStatus('Feedback could not be sent. Please try again.', true);
    }
});

thumbDownButton.addEventListener('click', () => {
    setFeedbackSelection('down');
    correctionPanel.hidden = false;
    setFeedbackStatus('Describe the issue and the mapping will be refined.');
    correctionText.focus();
    recordFeedback(false).catch((error) => {
        console.error('Feedback error:', error);
    });
});

submitCorrection.addEventListener('click', async () => {
    const correction = correctionText.value.trim();
    if (correction.length < 3) {
        setFeedbackStatus('Please briefly describe what needs changing.', true);
        correctionText.focus();
        return;
    }
    if (!currentResult || !currentLaxFile || !currentFlexedFile) {
        setFeedbackStatus('Upload and analyze both images before refining.', true);
        return;
    }

    const formData = new FormData();
    formData.append('lax_image', currentLaxFile);
    formData.append('flexed_image', currentFlexedFile);
    formData.append('analysis_json', JSON.stringify(currentResult.analysis));
    formData.append('analysis_id', currentResult.analysis_id);
    formData.append('feedback', correction);

    const originalLabel = submitCorrection.textContent;
    submitCorrection.disabled = true;
    thumbUpButton.disabled = true;
    thumbDownButton.disabled = true;
    submitCorrection.textContent = 'Refining...';
    setFeedbackStatus('Revising coordinates with your feedback...');

    try {
        const response = await fetch('/refine', {
            method: 'POST',
            body: formData,
        });
        const result = await readResponse(response);
        await displayResult(result);
        resetFeedback('Mapping updated - please review it again.');
    } catch (error) {
        console.error('Refinement error:', error);
        setFeedbackStatus(
            error.message || 'The mapping could not be refined. Please try again.',
            true,
        );
    } finally {
        submitCorrection.disabled = false;
        thumbUpButton.disabled = false;
        thumbDownButton.disabled = false;
        submitCorrection.textContent = originalLabel;
    }
});
