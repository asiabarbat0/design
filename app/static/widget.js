// DesignStream AI Widget - Enhanced Version
(function() {
    'use strict';
    
    console.log("DesignStream AI Widget loaded");
    
    document.addEventListener('DOMContentLoaded', function() {
        // Create enhanced widget button
        const widgetContainer = document.createElement('div');
        widgetContainer.id = 'designstream-widget';
        widgetContainer.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 10000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;
        
        const button = document.createElement('button');
        button.innerHTML = '<i class="fas fa-magic"></i> Design with AI';
        button.style.cssText = `
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 25px;
            border-radius: 25px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        `;
        
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 8px 25px rgba(102, 126, 234, 0.4)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 15px rgba(102, 126, 234, 0.3)';
        });
        
        button.onclick = function() {
            showDesignModal();
        };
        
        widgetContainer.appendChild(button);
        document.body.appendChild(widgetContainer);
        
        // Create design modal
        function showDesignModal() {
            const modal = document.createElement('div');
            modal.id = 'design-modal';
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                z-index: 10001;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            `;
            
            const modalContent = document.createElement('div');
            modalContent.style.cssText = `
                background: white;
                border-radius: 20px;
                padding: 30px;
                max-width: 500px;
                width: 100%;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            `;
            
            modalContent.innerHTML = `
                <div style="text-align: center; margin-bottom: 30px;">
                    <h2 style="color: #333; margin-bottom: 10px;">
                        <i class="fas fa-magic" style="color: #667eea;"></i> AI Design Studio
                    </h2>
                    <p style="color: #666;">Upload your room photo to get AI-powered design recommendations</p>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <label style="display: block; font-weight: 600; margin-bottom: 10px; color: #555;">
                        Room Photo
                    </label>
                    <div id="file-upload-area" style="
                        border: 2px dashed #ddd;
                        border-radius: 12px;
                        padding: 40px 20px;
                        text-align: center;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        background: #fafafa;
                    ">
                        <i class="fas fa-cloud-upload-alt" style="font-size: 3rem; color: #667eea; margin-bottom: 15px;"></i>
                        <div style="font-size: 1.1rem; color: #666; margin-bottom: 10px;">
                            Click to upload room photo
                        </div>
                        <div style="font-size: 0.9rem; color: #999;">
                            PNG, JPG up to 25MB
                        </div>
                    </div>
                    <input type="file" id="room-photo" accept="image/*" style="display: none;">
                </div>
                
                <div id="preview-area" style="display: none; margin-bottom: 20px;">
                    <img id="preview-image" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                </div>
                
                <div style="display: flex; gap: 15px;">
                    <button id="analyze-btn" style="
                        flex: 1;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border: none;
                        padding: 15px;
                        border-radius: 12px;
                        font-size: 16px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    " disabled>
                        <i class="fas fa-search"></i> Analyze Room
                    </button>
                    <button id="close-modal" style="
                        background: #f8f9fa;
                        color: #666;
                        border: 2px solid #e1e5e9;
                        padding: 15px 20px;
                        border-radius: 12px;
                        font-size: 16px;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    ">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <div id="loading" style="display: none; text-align: center; padding: 20px;">
                    <i class="fas fa-spinner fa-spin" style="font-size: 2rem; color: #667eea; margin-bottom: 10px;"></i>
                    <p>Analyzing your room...</p>
                </div>
                
                <div id="results" style="display: none; margin-top: 20px;">
                    <h3 style="color: #333; margin-bottom: 15px;">AI Recommendations</h3>
                    <div id="recommendations-list"></div>
                </div>
            `;
            
            modal.appendChild(modalContent);
            document.body.appendChild(modal);
            
            // Setup event listeners
            const fileUploadArea = modal.querySelector('#file-upload-area');
            const fileInput = modal.querySelector('#room-photo');
            const previewArea = modal.querySelector('#preview-area');
            const previewImage = modal.querySelector('#preview-image');
            const analyzeBtn = modal.querySelector('#analyze-btn');
            const closeBtn = modal.querySelector('#close-modal');
            const loading = modal.querySelector('#loading');
            const results = modal.querySelector('#results');
            const recommendationsList = modal.querySelector('#recommendations-list');
            
            fileUploadArea.addEventListener('click', () => fileInput.click());
            
            fileInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        previewImage.src = e.target.result;
                        previewArea.style.display = 'block';
                        analyzeBtn.disabled = false;
                        fileUploadArea.style.borderColor = '#667eea';
                        fileUploadArea.style.background = '#f0f4ff';
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            analyzeBtn.addEventListener('click', function() {
                const file = fileInput.files[0];
                if (!file) return;
                
                analyzeBtn.style.display = 'none';
                loading.style.display = 'block';
                
                const formData = new FormData();
                formData.append('room_photo', file);
                
                fetch('http://localhost:5002/widget/recommendations', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    results.style.display = 'block';
                    
                    if (data.recommendations && data.recommendations.length > 0) {
                        recommendationsList.innerHTML = data.recommendations.map(rec => `
                            <div style="
                                background: #f8f9fa;
                                border-radius: 8px;
                                padding: 15px;
                                margin-bottom: 10px;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            ">
                                <div>
                                    <strong>Variant ${rec.variant_id}</strong>
                                    <div style="color: #666; font-size: 0.9rem;">
                                        Score: ${rec.score.toFixed(2)}
                                    </div>
                                </div>
                                <button onclick="addToCart(${rec.variant_id})" style="
                                    background: #28a745;
                                    color: white;
                                    border: none;
                                    padding: 8px 15px;
                                    border-radius: 6px;
                                    cursor: pointer;
                                    font-size: 0.9rem;
                                ">
                                    Add to Cart
                                </button>
                            </div>
                        `).join('');
                    } else {
                        recommendationsList.innerHTML = '<p style="color: #666; text-align: center;">No recommendations found</p>';
                    }
                })
                .catch(error => {
                    loading.style.display = 'none';
                    analyzeBtn.style.display = 'block';
                    alert('Error analyzing room: ' + error.message);
                });
            });
            
            closeBtn.addEventListener('click', () => {
                document.body.removeChild(modal);
            });
            
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    document.body.removeChild(modal);
                }
            });
        }
        
        // Add to cart function
        window.addToCart = function(variantId) {
            fetch('/cart/add.js', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: variantId, quantity: 1 })
            })
            .then(response => response.json())
            .then(data => {
                alert('Added to cart successfully!');
            })
            .catch(error => {
                alert('Error adding to cart: ' + error.message);
            });
        };
        
        // Add powered by badge
        const badge = document.createElement('div');
        badge.innerHTML = 'Powered by <strong>DesignStream AI</strong>';
        badge.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(255,255,255,0.9);
            padding: 8px 15px;
            border-radius: 15px;
            font-size: 12px;
            color: #666;
            z-index: 10000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        `;
        document.body.appendChild(badge);
    });
})();