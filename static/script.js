// static/script.js
function sendMessage() {
    let input = document.getElementById("user-input");
    let message = input.value.trim();
    if (message === "") return;

    let chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<div class="user"><b>You:</b> ${escapeHtml(message)}</div>`;

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    })
    .then(res => res.json())
    .then(data => {
        chatBox.innerHTML += `<div class='bot'><b>Bot:</b><pre style="white-space:pre-wrap;">${escapeHtml(data.response)}</pre></div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    }).catch(err => {
        chatBox.innerHTML += `<div class='bot'>Error: ${err}</div>`;
    });

    input.value = "";
}

function escapeHtml(text) {
    var map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}
