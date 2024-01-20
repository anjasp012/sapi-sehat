document.addEventListener("DOMContentLoaded", function () {
  // Get all section elements with the class "section"
  var sections = document.querySelectorAll(".section");

  // Get the navbar element
  var navbar = document.querySelector(".navbar");

  // Function to update the navbar background color based on the visible section
  function updateNavbar() {
    var scrollPosition = window.scrollY;

    sections.forEach(function (section) {
      var sectionTop = section.offsetTop;
      var sectionBottom = sectionTop + section.offsetHeight;

      // Check if the scroll position is within the current section
      if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
        // Set the background color of the navbar
        navbar.style.backgroundColor = "#6faa02";
      }
    });
  }

  // Add a scroll event listener to update the navbar on scroll
  window.addEventListener("scroll", updateNavbar);

  // Initial update to set the navbar background color on page load
  updateNavbar();
});

// Menambahkan event saat dokumen selesai dimuat
document.addEventListener("DOMContentLoaded", function () {
  // Memilih elemen body
  var body = document.querySelector("body");

  // Membuat elemen button WhatsApp
  var whatsappButton = document.createElement("div");
  whatsappButton.innerHTML = '<a href="https://wa.me/+6285274253902" target="_blank"><i class="bi bi-whatsapp"></i></a>';
  whatsappButton.classList.add("whatsapp-float");

  // Menambahkan button WhatsApp ke dalam body
  body.appendChild(whatsappButton);
});
