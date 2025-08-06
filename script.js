document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const resultsSection = document.getElementById('results-section');
    const resultImage = document.getElementById('result-image');
    const classificationResults = document.getElementById('classification-results');
    const confusionMatrixImg = document.getElementById('confusion-matrix');
    const metricsChartImg = document.getElementById('metrics-chart');
    
    // Handle file upload
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file first');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading state
        classificationResults.innerHTML = `
            <div class="loading-state">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Processing image...</p>
                <p class="loading-note">This may take a few moments depending on image size</p>
            </div>
        `;
        resultsSection.style.display = 'block';
        
        // Clear previous results
        resultImage.src = '';
        confusionMatrixImg.src = '';
        metricsChartImg.src = '';
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Server error'); });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            if (data.type === 'image') {
                // Display processed image
                resultImage.src = data.result_img + '?' + new Date().getTime(); // Cache busting
                resultImage.onload = function() {
                    // Display detection results
                    if (data.detections && data.detections.length > 0) {
                        renderDetectionResults(data.detections);
                    } else {
                        classificationResults.innerHTML = `
                            <div class="no-detections">
                                <i class="fas fa-info-circle"></i>
                                <p>No detections found in the image</p>
                            </div>
                        `;
                    }
                    
                    // Display confusion matrix and metrics if available
                    if (data.confusion_matrix) {
                        confusionMatrixImg.src = `data:image/png;base64,${data.confusion_matrix}`;
                    }
                    
                    if (data.metrics_chart) {
                        metricsChartImg.src = `data:image/png;base64,${data.metrics_chart}`;
                    }
                };
            }
        })
        .catch(error => {
            console.error('Error:', error);
            classificationResults.innerHTML = `
                <div class="error-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Error: ${error.message}</p>
                    <button class="btn btn-retry" onclick="window.location.reload()">Try Again</button>
                </div>
            `;
        });
    });
    
    // Show file name when selected
    fileInput.addEventListener('change', function() {
        const fileName = this.files[0] ? this.files[0].name : 'No file chosen';
        document.querySelector('#image-upload-card button:first-of-type').textContent = fileName;
    });
    
    // Function to render detection results
    function renderDetectionResults(detections) {
    let html = `
        <div class="report-header">
            <h4><i class="fas fa-clipboard-list"></i> Detection Summary</h4>
            <p class="detection-count">Found ${detections.length} detection(s)</p>
        </div>
        <div class="detections-list">
    `;
    
    detections.forEach((detection, index) => {
        const finalClass = detection.final_class.replace('_', '-');
        const statusClass = `status-${finalClass}`;
        const yoloConfidence = (detection.yolo_confidence * 100).toFixed(1);
        const finalConfidence = (detection.final_confidence * 100).toFixed(1);
        
        html += `
            <div class="detection-item">
                <div class="detection-header">
                    <h5><i class="fas fa-user"></i> Detection ${index + 1}</h5>
                    <span class="status ${statusClass}">${detection.final_class.replace('_', ' ')}</span>
                </div>
                
                <div class="detection-details">
                    <p><strong>YOLO Initial Detection:</strong> 
                        ${detection.yolo_class.replace('_', ' ')} (${yoloConfidence}%)</p>
                    <p><strong>Final Classification:</strong> 
                        <span class="status ${statusClass}">${detection.final_class.replace('_', ' ')}</span> 
                        (${finalConfidence}%)</p>
                    <p><strong>Bounding Box:</strong> [${detection.box.map(x => x.toFixed(1)).join(', ')}]</p>
        `;
        
        // Add CNN results if available
        if (detection.cnn_probs) {
            html += `
                <div class="cnn-results">
                    <h6><i class="fas fa-brain"></i> CNN Probability Analysis</h6>
                    <div class="probability-bars">
                        <div class="probability-bar">
                            <label>Fall:</label>
                            <div class="bar-container">
                                <div class="bar-fill" style="width: ${detection.cnn_probs.fall}%; 
                                    background-color: #f44336;"></div>
                                <span>${detection.cnn_probs.fall}%</span>
                            </div>
                        </div>
                        <div class="probability-bar">
                            <label>Not Fall:</label>
                            <div class="bar-container">
                                <div class="bar-fill" style="width: ${detection.cnn_probs.not_fall}%; 
                                    background-color: #4CAF50;"></div>
                                <span>${detection.cnn_probs.not_fall}%</span>
                            </div>
                        </div>
                        <div class="probability-bar">
                            <label>Sitting:</label>
                            <div class="bar-container">
                                <div class="bar-fill" style="width: ${detection.cnn_probs.sitting}%; 
                                    background-color: #2196F3;"></div>
                                <span>${detection.cnn_probs.sitting}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += `
                </div>
            </div>
        `;
    });
    
    html += `</div>`; // Close detections-list
    classificationResults.innerHTML = html;
}
});