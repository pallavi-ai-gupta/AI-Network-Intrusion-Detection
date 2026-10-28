function confirmAction(button) {
    const message = button.dataset.confirm || "Continue?";
    return window.confirm(message);
}

document.addEventListener("DOMContentLoaded", function () {
    const forms = document.querySelectorAll("form[data-loading]");

    forms.forEach(function (form) {
        form.addEventListener("submit", function () {
            const button = form.querySelector('button[type="submit"]');
            if (button) {
                button.disabled = true;
                button.textContent = "Please wait...";
            }
        });
    });

    const fileInputs = document.querySelectorAll('input[type="file"]');

    fileInputs.forEach(function (input) {
        input.addEventListener("change", function () {
            if (input.files.length > 0) {
                const file = input.files[0];
                if (!file.name.toLowerCase().endsWith(".csv")) {
                    alert("Please choose a CSV file.");
                    input.value = "";
                }
            }
        });
    });
});