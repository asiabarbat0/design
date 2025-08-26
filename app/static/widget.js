document.addEventListener('DOMContentLoaded', function() {
    const button = document.createElement('button');
    button.innerText = 'Design with my room';
    button.onclick = function() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = function(event) {
            const file = event.target.files[0];
            const formData = new FormData();
            formData.append('room_photo', file);
            fetch('http://localhost:5001/widget/recommendations', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const recsDiv = document.createElement('div');
                data.recommendations.forEach(rec => {
                    const item = document.createElement('div');
                    item.innerText = `Variant ${rec.variant_id} (Score: ${rec.score})`;
                    const atcButton = document.createElement('button');
                    atcButton.innerText = 'Add to Cart';
                    atcButton.onclick = () => {
                        fetch('/cart/add.js', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ id: rec.variant_id, quantity: 1 })
                        });
                    };
                    item.appendChild(atcButton);
                    recsDiv.appendChild(item);
                });
                document.body.appendChild(recsDiv);
            });
        };
        input.click();
    };
    document.body.appendChild(button);
    const badge = document.createElement('div');
    badge.innerText = 'Powered by DesignStreamAI';
    badge.style.fontSize = '12px';
    document.body.appendChild(badge);
});