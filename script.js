// Handle file selection and preview
function handleFileChange(event) {
  const files = event.target.files;
  previewFiles(files);
}

function previewFiles(files) {
  const previewContainer = document.getElementById('preview');
  const dropAreaLabel = document.getElementById('drop-area-label');

  // Hide label text when an image is added
  dropAreaLabel.style.display = 'none';

  // Clear previous previews
  previewContainer.innerHTML = '';

  // Loop through files and add each to preview
  Array.from(files).forEach(file => {
      if (file.type.startsWith('image/')) {
          const reader = new FileReader();
          reader.onload = function(event) {
              const imgContainer = document.createElement('div');
              imgContainer.className = 'preview-item'; // Updated class for styling

              const img = document.createElement('img');
              img.src = event.target.result;
              img.className = 'preview-image';

              // Create remove button
              const removeButton = document.createElement('button');
              removeButton.className = 'remove-button';
              removeButton.innerHTML = '&times;';
              removeButton.onclick = () => {
                  imgContainer.remove();
                  if (previewContainer.childElementCount === 0) {
                      dropAreaLabel.style.display = 'flex';
                  }
              };

              // Append elements
              imgContainer.appendChild(img);
              imgContainer.appendChild(removeButton);
              previewContainer.appendChild(imgContainer);
          };
          reader.readAsDataURL(file);
      }
  });
}

// Drag-and-drop events
const dropArea = document.getElementById('drop-area');

dropArea.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropArea.classList.add('highlight');
});

dropArea.addEventListener('dragleave', () => {
  dropArea.classList.remove('highlight');
});

dropArea.addEventListener('drop', (event) => {
  event.preventDefault();
  dropArea.classList.remove('highlight');
  const files = event.dataTransfer.files;
  previewFiles(files);
});

function applyFilter() {
  const previewContainer = document.getElementById('preview');
  const imgElement = previewContainer.querySelector('img'); // Get the first image

  if (!imgElement) {
    alert('Error: Please upload an image before applying the filter.');
    return;
  }

  if (imgElement) {
    // Store the image data URL in session storage
    sessionStorage.setItem('initialImage', imgElement.src);

    // Call the Gaussian noise function in MATLAB (server-side logic)
    fetch('processGaussianNoise', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ imagePath: 'uploaded_image.jpg', mean: 0, variance: 25 }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Save the noisy image to session storage
        sessionStorage.setItem('noisyImage', data.noisyImage);

        // Redirect to the results page
        window.location.href = 'results.html';
      })
      .catch((error) => {
        console.error('Error processing Gaussian noise:', error);
        alert('An error occurred while processing the Gaussian noise.');
      });
  }
}

function loadResults() {
  const initialImage = sessionStorage.getItem('initialImage');
  const noisyImage = sessionStorage.getItem('noisyImage');

  if (initialImage) {
    const initialImageSection = document.querySelector('.grid .section-title:nth-child(1)');
    initialImageSection.insertAdjacentHTML(
      'afterend',
      `<img src="${initialImage}" alt="Initial Image" class="results-image">`
    );
  }

  if (noisyImage) {
    const blurredImageSection = document.querySelector('.grid .section-title:nth-child(2)');
    blurredImageSection.insertAdjacentHTML(
      'afterend',
      `<img src="${noisyImage}" alt="Blurred Image" class="results-image">`
    );
  }
}

function goBack() {
  window.location.href = 'index.html'; // Replace with the actual path to your index page
}
