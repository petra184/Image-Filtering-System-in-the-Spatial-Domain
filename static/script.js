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
  const processingChannel = document.querySelector('input[name="channel"]:checked')?.value;
  const noiseModeling = document.querySelector('input[name="noise"]:checked')?.value;
  const filter = document.querySelector('input[name="filter"]:checked')?.value;

  const previewContainer = document.getElementById('preview');
  const imgElement = previewContainer.querySelector('img');
  
  if (!imgElement) {
    alert('Error: Please upload an image before applying the filter.');
    return;
  }

  const imageSrc = imgElement.src;
  if (!imageSrc.startsWith('data:image/')) {
    alert('Only base64 encoded images are supported');
    return;
  }

  // Store the image and filter data in sessionStorage
  sessionStorage.setItem('initialImage', imageSrc); // Store original image
  sessionStorage.setItem('channel', processingChannel); 
  sessionStorage.setItem('noise', noiseModeling);
  sessionStorage.setItem('filter_tu', filter);
    
  window.location.href = '/results';
}

function loadResults() {
  const initialImage = sessionStorage.getItem('initialImage');
  const noisyImage = sessionStorage.getItem('noisyImage');
  const channel = sessionStorage.getItem('channel');
  const noise = sessionStorage.getItem('noise');
  const filter_tu = sessionStorage.getItem('filter_tu');

  // Insert the initial image
  const initialImageSection = document.querySelector('.grid .section-title:nth-child(1)');
  if (initialImageSection && initialImage) {
    initialImageSection.insertAdjacentHTML(
      'afterend',
      `<img src="${initialImage}" alt="Initial Image" class="results-image">`
    );
  }

  const data = {
    image: initialImage,
    processingChannel: channel,
    noiseModeling: noise,
    filter: filter_tu,
  };

  fetch("http://127.0.0.1:5000/image_processing", {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('Failed to fetch results from the backend.');
      }
      return response.json();
    })
    .then(data => {
      const noisyImage = data.noisy_image;
      let cValue = data.c_value;

    // Round `cValue` to two decimal places
      if (cValue !== undefined && cValue !== null) {
        cValue = parseFloat(cValue).toFixed(2);
      }

      // Insert the noisy (blurred) image
      const sectionTitles = document.querySelectorAll('.section-title');
      const blurredImageSection = sectionTitles[1]; // Index corresponds to "Noisy Image"
      if (blurredImageSection && noisyImage) {
        blurredImageSection.insertAdjacentHTML(
          'afterend',
          `<img src="${noisyImage}" alt="Noisy Image" class="results-image">`
        );
      }

      const statsSection = sectionTitles[3]; // Index corresponds to "Statistical Characteristics"
      if (statsSection) {
        statsSection.insertAdjacentHTML(
          'afterend',
          `<ul class="results-stats">
            <li><strong>Processing Channel:</strong> ${channel || 'Not specified'}</li>
            <li><strong>Filter:</strong> ${filter_tu || 'Not specified'}</li>
            <li><strong>Noise Modeling:</strong> ${noise || 'Not specified'}</li>
            <li><strong> ${noise === 'Gaussian' ? 'Noise Corruption Coefficient' : noise === 'Impulse' ? 'Corruption Rate' : 'Noise Corruption Coefficient/Corruption Rate'}:
            </strong> ${cValue || 'Not available'}
            </li>
          </ul>`
        );
      }
      
    })
    .catch(error => {
      console.error('Error fetching the noisy image:', error);
      alert('An error occurred while processing the image.');
    });
}


function goBack() {
  window.location.href = 'http://127.0.0.1:5000/';
}