const express = require('express');
const cors = require('cors');
const { MongoClient } = require('mongodb');

const app = express();
const port = 8000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB connection
const uri = 'mongodb://localhost:27017';
const client = new MongoClient(uri);
let db;

client.connect()
  .then(() => {
    db = client.db('portfolio_db');
    console.log('Connected to MongoDB');

    // Start server only after successful DB connection
    app.listen(port, () => {
      console.log(`Server is running on http://localhost:${port}`);
    });
  })
  .catch(err => {
    console.error('Failed to connect to MongoDB', err);
    process.exit(1);
  });
  app.get('/portfolios', async (req, res) => {
    try {
      const { 
        min_score, 
        max_score, 
        color, 
        keyword, 
        threshold, 
        page = 1, 
        limit = 12, 
        min_match_score = 0,
        text_density_min, 
        text_density_max, 
        color_variance_min, 
        color_variance_max, 
        layout_complexity_min, 
        layout_complexity_max,
        category
      } = req.query;
  
      const filter = {};
  
      // Filter by rating range
      if (min_score && max_score) {
        filter.rating = { $gte: parseInt(min_score), $lte: parseInt(max_score) };
      }
  
      // Filter by keyword in URL
      if (keyword) {
        filter.url = { $regex: keyword, $options: 'i' };
      }
  
      // Filter by text density range
      if (text_density_min && text_density_max) {
        filter.text_density = { $gte: parseFloat(text_density_min), $lte: parseFloat(text_density_max) };
      }
  
      // Filter by color variance range
      if (color_variance_min && color_variance_max) {
        filter.color_variance = { $gte: parseFloat(color_variance_min), $lte: parseFloat(color_variance_max) };
      }
  
      // Filter by layout complexity range
      if (layout_complexity_min && layout_complexity_max) {
        filter.layout_complexity = { $gte: parseFloat(layout_complexity_min), $lte: parseFloat(layout_complexity_max) };
      }
  
      // Filter by category
      if (category) {
        filter.category = category;
      }
  
      let portfolios = await db.collection('portfolios').find(filter).toArray();
  
      // Handle color profile filtering if needed
      let userProfile = [];
      if (color) {
        userProfile = JSON.parse(color);
      }
  
      const thresholdValue = threshold ? parseFloat(threshold) : 50;
      const minMatchScore = min_match_score ? parseFloat(min_match_score) : 0;
  
      portfolios = portfolios.map(portfolio => {
        portfolio.score = calculateCompositeScore(userProfile, portfolio, thresholdValue);
        return portfolio;
      });
  
      // Filter portfolios by composite score
      portfolios = portfolios.filter(portfolio => portfolio.score >= minMatchScore);
  
      // Sort portfolios by composite score
      portfolios.sort((a, b) => b.score - a.score);
  
      // Pagination
      const startIndex = (page - 1) * limit;
      const endIndex = startIndex + parseInt(limit);
      const paginatedPortfolios = portfolios.slice(startIndex, endIndex);
  
      res.json({ portfolios: paginatedPortfolios, total: portfolios.length });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: err.message });
    }
  });  

app.delete('/portfolio', async (req, res) => {
  try {
    const { url } = req.query;
    if (!url) {
      return res.status(400).json({ error: 'URL parameter is required' });
    }

    const result = await db.collection('portfolios').deleteOne({ url });
    if (result.deletedCount === 0) {
      return res.status(404).json({ error: 'Portfolio not found' });
    }

    res.status(200).json({ message: 'Portfolio deleted successfully' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.delete('/portfolios', async (req, res) => {
  try {
    await db.collection('portfolios').deleteMany({});
    res.status(200).json({ message: 'All portfolios deleted successfully' });
  } catch (err) {
    son
    res.status(500).json({ error: err.message });
  }
});

function colorDistance(c1, c2) {
  return Math.sqrt(
    Math.pow(c1[0] - c2[0], 2) +
    Math.pow(c1[1] - c2[1], 2) +
    Math.pow(c1[2] - c2[2], 2)
  );
}

function colorDistance(c1, c2) {
  return Math.sqrt(Math.pow(c1[0] - c2[0], 2) + Math.pow(c1[1] - c2[1], 2) + Math.pow(c1[2] - c2[2], 2));
}

function calculateProfileScore(userProfile, actualProfile, threshold) {
  let score = 0.0;

  userProfile.forEach((userColor, i) => {
    let minDistance = Infinity;
    const actualColor = actualProfile[i];

    if (i < actualProfile.length) {
      const distance = colorDistance(userColor.color, actualColor.color);
      if (distance < threshold) {
        minDistance = distance;
      }
    }

    const orderDifference = Math.abs(i - actualProfile.findIndex(profileColor => colorDistance(profileColor.color, userColor.color) === 0));
    score += (threshold - minDistance) * (1.0 / (1.0 + orderDifference));
  });

  return score;
}

function calculateCompositeScore(userProfile, portfolio, threshold) {
  const colorScore = userProfile.length > 0 ? calculateProfileScore(userProfile, portfolio.color_profile, threshold) : 0;
  const ratingScore = portfolio.rating ? portfolio.rating : 0;

  return colorScore * 0.7 + ratingScore * 0.3;
}

// Serve static files (e.g., frontend files)
app.use(express.static('public'));
