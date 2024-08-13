
document.addEventListener("DOMContentLoaded", function() {
  const form = document.getElementById("search-form");
  const portfolioList = document.getElementById("portfolio-list");
  const deleteAllPortfoliosBtn = document.getElementById("delete-all-portfolios");
  const firstPageBtn = document.getElementById("first-page");
  const prevPageBtn = document.getElementById("prev-page");
  const nextPageBtn = document.getElementById("next-page");
  const lastPageBtn = document.getElementById("last-page");
  const goToPageInput = document.getElementById("go-to-page-input");
  const goToPageButton = document.getElementById("go-to-page-button");
  const pageInfo = document.getElementById("page-info");
  const colorPicker = document.querySelector(".color-picker");
  const addColorBtn = document.getElementById("add-color");
  const colorList = document.getElementById("color-list");
  const thresholdSlider = document.getElementById("threshold");
  const thresholdValue = document.getElementById("thresholdValue");

  function updateSlider(minRange, maxRange, minValue, maxValue, sliderTrack) {
      console.log("update slider")
      const min = parseInt(minRange.value);
      const max = parseInt(maxRange.value);

      if (min > max) {
          minRange.value = max;
      }

      if (max < min) {
          maxRange.value = min;
      }

      minValue.textContent = minRange.value;
      maxValue.textContent = maxRange.value;

      const percentMin = ((minRange.value - minRange.min) / (minRange.max - minRange.min)) * 100;
      const percentMax = ((maxRange.value - maxRange.min) / (maxRange.max - maxRange.min)) * 100;

      sliderTrack.style.left = percentMin + '%';
      sliderTrack.style.width = (percentMax - percentMin) + '%';
  }

  const minRangeRating = document.getElementById('min-rating');
  const maxRangeRating = document.getElementById('max-rating');
  const minValueRating = document.getElementById('min-value-rating');
  const maxValueRating = document.getElementById('max-value-rating');
  const sliderTrackRating = document.querySelector('.slider-track-rating');

  minRangeRating.addEventListener('input', () => updateSlider(minRangeRating, maxRangeRating, minValueRating, maxValueRating, sliderTrackRating));
  maxRangeRating.addEventListener('input', () => updateSlider(minRangeRating, maxRangeRating, minValueRating, maxValueRating, sliderTrackRating));
  updateSlider(minRangeRating, maxRangeRating, minValueRating, maxValueRating, sliderTrackRating); // Initialize the slider position

  const minRangeText = document.getElementById('min-text');
  const maxRangeText = document.getElementById('max-text');
  const minValueText = document.getElementById('min-value-text');
  const maxValueText = document.getElementById('max-value-text');
  const sliderTrackText = document.querySelector('.slider-track-text');

  minRangeText.addEventListener('input', () => updateSlider(minRangeText, maxRangeText, minValueText, maxValueText, sliderTrackText));
  maxRangeText.addEventListener('input', () => updateSlider(minRangeText, maxRangeText, minValueText, maxValueText, sliderTrackText));
  updateSlider(minRangeText, maxRangeText, minValueText, maxValueText, sliderTrackText); // Initialize the slider position


  const minRangeVariance = document.getElementById('min-variance');
  const maxRangeVariance = document.getElementById('max-variance');
  const minValueVariance = document.getElementById('min-value-variance');
  const maxValueVariance = document.getElementById('max-value-variance');
  const sliderTrackVariance = document.querySelector('.slider-track-variance');

  minRangeVariance.addEventListener('input', () => updateSlider(minRangeVariance, maxRangeVariance, minValueVariance, maxValueVariance, sliderTrackVariance));
  maxRangeVariance.addEventListener('input', () => updateSlider(minRangeVariance, maxRangeVariance, minValueVariance, maxValueVariance, sliderTrackVariance));
  updateSlider(minRangeVariance, maxRangeVariance, minValueVariance, maxValueVariance, sliderTrackVariance); // Initialize the slider position


  const minRangeLayout = document.getElementById('min-layout');
  const maxRangeLayout = document.getElementById('max-layout');
  const minValueLayout = document.getElementById('min-value-layout');
  const maxValueLayout = document.getElementById('max-value-layout');
  const sliderTrackLayout = document.querySelector('.slider-track-layout');

  minRangeLayout.addEventListener('input', () => updateSlider(minRangeLayout, maxRangeLayout, minValueLayout, maxValueLayout, sliderTrackLayout));
  maxRangeLayout.addEventListener('input', () => updateSlider(minRangeLayout, maxRangeLayout, minValueLayout, maxValueLayout, sliderTrackLayout));
  updateSlider(minRangeLayout, maxRangeLayout, minValueLayout, maxValueLayout, sliderTrackLayout); // Initialize the slider position



  let currentPage = 1;
  let totalPages = 1;
  const limit = 12;
  const maxColors = 5;
  let selectedColors = [];

  if (deleteAllPortfoliosBtn) {
    deleteAllPortfoliosBtn.addEventListener("click", async function() {
      if (confirm("Are you sure you want to delete all portfolios?")) {
        await fetch('http://localhost:8000/portfolios', { method: 'DELETE' });
        fetchPortfolios();
      }
    });  
  }

  async function deletePortfolio(url) {
    if (confirm("Are you sure you want to delete this portfolio?")) {
      await fetch(`http://localhost:8000/portfolio?url=${encodeURIComponent(url)}`, { method: 'DELETE' });
      fetchPortfolios();
    }
  }

  form.addEventListener("submit", function(event) {
    event.preventDefault();
    currentPage = 1;
    fetchPortfolios();
  });

  addColorBtn.addEventListener("click", function() {
    if (selectedColors.length < maxColors) {
      const color = colorPicker.value;
      selectedColors.push(color);
      updateColorList();
    } else {
      alert("You can only select up to 5 colors.");
    }
  });

  firstPageBtn.addEventListener("click", function() {
    if (currentPage > 1) {
      currentPage = 1;
      fetchPortfolios();
    }
  });

  prevPageBtn.addEventListener("click", function() {
    if (currentPage > 1) {
      currentPage -= 1;
      fetchPortfolios();
    }
  });

  nextPageBtn.addEventListener("click", function() {
    if (currentPage < totalPages) {
      currentPage += 1;
      fetchPortfolios();
    }
  });

  lastPageBtn.addEventListener("click", function() {
    if (currentPage < totalPages) {
      currentPage = totalPages;
      fetchPortfolios();
    }
  });

  // Event listener for "Go to Page" button
  goToPageButton.addEventListener("click", function() {
    const requestedPage = parseInt(goToPageInput.value, 10);
    if (requestedPage >= 1 && requestedPage <= totalPages) {
      currentPage = requestedPage;
      fetchPortfolios();
    } else {
      alert(`Please enter a valid page number between 1 and ${totalPages}.`);
    }
  });


  // Update the threshold value display when the slider moves
  thresholdSlider.addEventListener("input", function() {
    thresholdValue.textContent = thresholdSlider.value;
  });

  function updateColorList() {
    colorList.innerHTML = '';
    selectedColors.forEach((color, index) => {
      const li = document.createElement("li");
      const colorBox = document.createElement("div");
      colorBox.className = "color-box";
      colorBox.style.backgroundColor = color;
      const colorText = document.createElement("span");
      colorText.textContent = color;
      const removeBtn = document.createElement("button");
      removeBtn.textContent = "x";
      removeBtn.addEventListener("click", function() {
        selectedColors.splice(index, 1);
        updateColorList();
      });

      li.appendChild(colorBox);
      li.appendChild(colorText);
      li.appendChild(removeBtn);
      colorList.appendChild(li);
    });

    // Update hidden input with selected colors in JSON format
    document.getElementById("color").value = JSON.stringify(selectedColors.map((color, index) => ({
      color: hexToRgb(color),
      order: index
    })));
  }

  async function fetchPortfolios() {
    const minScore = document.getElementById("min-rating").value;
    const maxScore = document.getElementById("max-rating").value;
    const color = document.getElementById("color").value;
    const keyword = document.getElementById("keyword").value;
    const threshold = document.getElementById("threshold").value;
  
    // Capture new filter values
    const category = document.getElementById('category').value;
    console.log(category);

    const minText = document.getElementById("min-text").value/100.0;
    const maxText = document.getElementById("max-text").value/100.0;

    const minVariance = document.getElementById("min-variance").value;
    const maxVariance = document.getElementById("max-variance").value;

    const minLayout = document.getElementById("min-layout").value/100.0;
    const maxLayout = document.getElementById("max-layout").value/100.0;

    const params = new URLSearchParams({
      min_score: minScore,
      max_score: maxScore,
      color: color,
      keyword: keyword,
      threshold: threshold,
      text_density_min: minText,
      text_density_max: maxText, 
      color_variance_min: minVariance,
      color_variance_max: maxVariance,
      layout_complexity_min: minLayout,
      layout_complexity_max: maxLayout,
      category: category,
      page: currentPage,
      limit: limit
    });

    const response = await fetch(`http://localhost:8000/portfolios?${params.toString()}`);
    const data = await response.json();
    displayPortfolios(data.portfolios);

    totalPages = Math.ceil(data.total / limit);
    pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
  }

  function displayPortfolios(portfolios) {
    portfolioList.innerHTML = '';
    portfolios.forEach(portfolio => {
      const li = document.createElement("li");
      const link = document.createElement("a");
      link.href = portfolio.url;
      link.textContent = portfolio.url;
      const score = document.createElement("p");
      score.innerHTML = `Score: ${portfolio.rating}`;

      const colorProfileDiv = document.createElement("div");
      colorProfileDiv.className = "color-profile";

      portfolio.color_profile.forEach(colorProfile => {
        const colorTile = document.createElement("div");
        colorTile.className = "color-tile";
        colorTile.style.backgroundColor = `rgb(${colorProfile.color[0]}, ${colorProfile.color[1]}, ${colorProfile.color[2]})`;
        const colorPercentage = document.createElement("span");
        colorPercentage.textContent = `${(colorProfile.percentage * 100).toFixed(1)}%`;
        const colorTileDiv = document.createElement("div");
        colorTileDiv.appendChild(colorTile);
        colorTileDiv.appendChild(colorPercentage);
        colorProfileDiv.appendChild(colorTileDiv);
      });

      const screenshot = document.createElement("img");
      screenshot.src = `../${portfolio.screenshot}`;
      console.log("screenshot.src: " + screenshot.src);
      screenshot.alt = "Screenshot";
      screenshot.className = "screenshot";

      const deleteBtn = document.createElement("button");
      deleteBtn.textContent = "Delete";
      deleteBtn.addEventListener("click", function() {
        deletePortfolio(portfolio.url);
      });

      const bottomDiv = document.createElement("div");
      bottomDiv.className = "bottom-div";
      bottomDiv.appendChild(score);
      bottomDiv.appendChild(deleteBtn);

      li.appendChild(screenshot);
      li.appendChild(link);
      li.appendChild(colorProfileDiv);
      li.appendChild(bottomDiv);
      portfolioList.appendChild(li);
    });
  }

  function hexToRgb(hex) {
    const bigint = parseInt(hex.slice(1), 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return [r, g, b];
  }
});