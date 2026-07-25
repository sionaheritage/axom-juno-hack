document.getElementById('uploadForm').onsubmit = async (e) => {
    e.preventDefault();

    const laxFile = document.getElementById('lax').files[0];
    const flexedFile = document.getElementById('flexed').files[0];

    const formData = new FormData();
    formData.append('lax_image', laxFile);
    formData.append('flexed_image', flexedFile);

    document.getElementById('loadingState').style.display = 'block';
    document.getElementById('resultContainer').style.display = 'none';
    document.getElementById('errorState').style.display = 'none';

    try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });

        if (!response.ok) {
            const contentType = response.headers.get('content-type') || '';
            const err = contentType.includes('application/json')
                ? await response.json()
                : { detail: await response.text() };
            throw new Error(err.detail || `Request failed with HTTP ${response.status}.`);
        }

        const blob = await response.blob();
        document.getElementById('resultImage').src = URL.createObjectURL(blob);
        document.getElementById('resultContainer').style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
        const errorState = document.getElementById('errorState');
        errorState.textContent = error.message || 'Connection lost. Check that the API server is running.';
        errorState.style.display = 'block';
    } finally {
        document.getElementById('loadingState').style.display = 'none';
    }
};
