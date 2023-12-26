function copytext(inputId) {
  // Get the text field
  var copyText = document.getElementById(inputId);

  // Select the text field
  copyText.select();
  copyText.setSelectionRange(0, 99999); // For mobile devices

   // Copy the text inside the text field
  navigator.clipboard.writeText(copyText.value);
}

  // Function to be executed after page reloads
  function afterPageLoad() {
    // Check if the #section2 element exists
    var section2Element = document.getElementById('proglang');

    if (section2Element) {
      // Perform your action here
      window.location.href = '#proglang';
      section2Element.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest', scrollDuration: 100 });
    }
  }

  // Attach the afterPageLoad function to the window.onload event
  window.onload = afterPageLoad;

function closeOverlay(event) {
    // Check if the clicked element is the overlay itself (not its children)
    if (event.target.classList.contains('overlay')) {
        window.location.href = '#'; // Close the overlay or add your desired behavior
    }
}

// Add this event listener to prevent closing the overlay when clicking on its children
document.getElementById('popup1').addEventListener('click', closeOverlay);



const textarea = document.getElementById('user_input');

const btn = document.getElementById('clear');

btn.addEventListener('click', function handleClick() {
  // ️ log value before clearing it
  console.log(textarea.value);

  //️ clear textarea value
  textarea.value = '';

  event.preventDefault();
});

function initializeSlider(sliderClass) {
  let mouseDown = false;
  let startX, scrollLeft;
  const slider = document.querySelector(sliderClass);

  const startDragging = (e) => {
    mouseDown = true;
    startX = e.pageX - slider.offsetLeft;
    scrollLeft = slider.scrollLeft;
  }

  const stopDragging = () => {
    mouseDown = false;
  }

  const move = (e) => {
    e.preventDefault();
    if (!mouseDown) {
      return;
    }
    const x = e.pageX - slider.offsetLeft;
    const scroll = x - startX;
    slider.scrollLeft = scrollLeft - scroll;
  }

  // Add the event listeners
  slider.addEventListener('mousemove', move, false);
  slider.addEventListener('mousedown', startDragging, false);
  slider.addEventListener('mouseup', stopDragging, false);
  slider.addEventListener('mouseleave', stopDragging, false);
}

initializeSlider('.div_programming');
initializeSlider('.div_projects');
initializeSlider('.div_paper');
