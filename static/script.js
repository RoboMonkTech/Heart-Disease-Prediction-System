// ======================================
// HEART DISEASE PREDICTION SYSTEM
// Professional JavaScript
// ======================================

const form = document.getElementById("predictionForm");
const button = document.querySelector(".button-area button");

if (form) {

    form.addEventListener("submit", function () {

        button.disabled = true;

        button.innerHTML = `
            <i class="fa-solid fa-spinner fa-spin"></i>
            Analyzing Patient Data...
        `;

    });

}

// ======================================
// Scroll to Result
// ======================================

window.addEventListener("load", function () {

    const result = document.querySelector(".result-card");

    if (result) {

        result.scrollIntoView({

            behavior: "smooth",

            block: "start"

        });

    }

});

// ======================================
// Input Validation
// ======================================

const numberInputs = document.querySelectorAll("input[type='number']");

numberInputs.forEach(input => {

    input.addEventListener("input", () => {

        if (input.value < 0) {

            input.value = "";

        }

    });

});

// ======================================
// Button Hover Animation
// ======================================

if (button) {

    button.addEventListener("mouseenter", () => {

        button.style.transform = "translateY(-3px) scale(1.02)";

    });

    button.addEventListener("mouseleave", () => {

        button.style.transform = "translateY(0px) scale(1)";

    });

}