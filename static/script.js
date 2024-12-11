function handleFileChange(event) {
  const files = event.target.files;
  previewFiles(files);
}

function previewFiles(files, fileInput) {
  const previewContainer = document.getElementById('preview');
  const dropAreaLabel = document.getElementById('drop-area-label');

  dropAreaLabel.style.display = 'none';
  previewContainer.innerHTML = '';

  // Loop through files and add each to preview
  Array.from(files).forEach((file, index) => {
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = function (event) {
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

          // If no more children, reset the drop area label and clear the input
          if (previewContainer.childElementCount === 0) {
            dropAreaLabel.style.display = 'flex';
            fileInput.value = ''; // Reset the file input
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
      if (cValue !== undefined && cValue !== null) {
        cValue = parseFloat(cValue).toFixed(2);
      }
      const filteredImage = data.filtered_image;
      rmse = data.rmse;
      psnr = data.psnr;
      stats_initial = data.stats_initial;
      stats_noisy = data.stats_noisy;
      stats_filtered = data.stats_filtered;

      const sectionTitles = document.querySelectorAll('.section-title');
      const blurredImageSection = sectionTitles[1]; // Index corresponds to "Noisy Image"
      if (blurredImageSection && noisyImage) {
        blurredImageSection.insertAdjacentHTML(
          'afterend',
          `<img src="${noisyImage}" alt="Noisy Image" class="results-image">`
        );
      }

      const filteredImageSection = sectionTitles[2]; // Index corresponds to "Filtered Image"
      if (filteredImageSection && filteredImage) {
        filteredImageSection.insertAdjacentHTML(
          'afterend',
          `<img src="${filteredImage}" alt="Filtered Image" class="results-image">`
        );
      }

      const statsTable = document.querySelector('#statistical-characteristics-table-container tbody');
      if (statsTable) {
          const statsRow = statsTable.rows[0];
          if (statsRow) {
              statsRow.cells[0].textContent = channel || 'Not specified';

              statsRow.cells[1].textContent = filter_tu || 'Not specified';

              statsRow.cells[2].textContent = noise || 'Not specified';

              const corruptionRateLabel = noise === 'Gaussian' 
                  ? 'Noise Corruption Coefficient' 
                  : (noise === 'Impulse' ? 'Corruption Rate' : 'Noise Corruption Coefficient/Corruption Rate');
              statsRow.cells[3].textContent = cValue || 'Not available';
          }
      }


        const metricsTable = document.querySelector('#metrics-table-container tbody');
        if (metricsTable) {
        // Populate the RMSE row
          const rmseRow = metricsTable.rows[0];
          if (rmseRow && rmse) {
              rmseRow.cells[1].textContent = rmse.R || '-'; // R column
              rmseRow.cells[2].textContent = rmse.G || '-'; // G column
              rmseRow.cells[3].textContent = rmse.B || '-'; // B column
              rmseRow.cells[4].textContent = rmse.Y || '-'; // Y (luma) column
              rmseRow.cells[5].textContent = rmse.Combined || '-'; // Combined column
          }

          // Populate the PSNR row
          const psnrRow = metricsTable.rows[1];
          if (psnrRow && psnr) {
              psnrRow.cells[1].textContent = psnr.R || '-'; // R column
              psnrRow.cells[2].textContent = psnr.G || '-'; // G column
              psnrRow.cells[3].textContent = psnr.B || '-'; // B column
              psnrRow.cells[4].textContent = psnr.Y || '-'; // Y (luma) column
              psnrRow.cells[5].textContent = psnr.Combined || '-'; // Combined column
          }
        }
        
        populateTable(stats_initial, 'initial-image-table-container');
        populateTable(stats_noisy, 'noisy-image-table-container');
        populateTable(stats_filtered, 'filtered-image-table-container');
    })
    .catch(error => {
      console.error('Error fetching the noisy image:', error);
      alert('An error occurred while processing the image.');
    });
}

function populateTable(stats, tableId) {
  // Get the table body for the specified table ID
  const tableBody = document.querySelector(`#${tableId} tbody`);
  if (!tableBody || !stats) {
      console.error('Invalid table ID or stats object');
      return;
  }

  // Define the order of metrics in the table
  const metricsOrder = ['Min', 'Max', 'Mean', 'Standard Deviation', 'Variance', 'SNR'];

  // Iterate over each row in the table
  metricsOrder.forEach((metric, rowIndex) => {
      const row = tableBody.rows[rowIndex];
      if (row) {
          // Populate each cell for the metric (R, G, B columns)
          ['R', 'G', 'B'].forEach((channel, colIndex) => {
              const cell = row.cells[colIndex + 1]; // +1 because the first cell is the metric name
              if (cell) {
                  cell.textContent = stats[channel]?.[metric] ?? '-';
              }
          });
      }
  });
}


function goBack() {
  window.location.href = 'http://127.0.0.1:5000/';
}