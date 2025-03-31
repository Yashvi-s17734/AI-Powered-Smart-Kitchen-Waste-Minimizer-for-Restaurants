const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const inventory = [
  { id: 1, name: 'Tomato', quantity: 10, expires: '2025-04-05' },
  { id: 2, name: 'Chicken', quantity: 5, expires: '2025-04-01' },
  { id: 4, name: 'Carrot', quantity: 6, expires: '2025-04-08' },
];

app.get('/api/inventory', (req, res) => {
  console.log('Serving inventory');
  res.json(inventory);
});
const axiosWithRetry = async (url, retries = 3, delay = 1000) => {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await axios.get(url, { 
        timeout: 5000,
        headers: { 'Content-Type': 'application/json' }  // Explicit headers
      });
      console.log(`Successfully connected to ${url}`);
      return response;
    } catch (error) {
      console.error(`Attempt ${i + 1}/${retries} failed for ${url}:`, error.message);
      if (i === retries - 1) {
        throw new Error(`Failed to connect to Python backend: ${error.message}`);
      }
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
};

app.get('/api/recommend_dishes', async (req, res) => {
  try {
    console.log('Fetching recommendations from Python...');
    const response = await axiosWithRetry('http://localhost:5001/api/recommend_dishes', {
      timeout: 20000 // Match frontend timeout
    });
    
    // Ensure response is properly formatted
    if (!Array.isArray(response.data)) {
      throw new Error("Invalid response format from Python");
    }

    console.log('Python response:', response.data);
    res.json(response.data); // Forward exact response
  } catch (error) {
    console.error('Recommend error:', error.message);
    res.status(500).json({ 
      error: 'Failed to fetch recommendations',
      details: error.message
    });
  }
});
app.get('/api/optimize_cost', (req, res) => {
  console.log('Optimizing costs');
  const costs = { Tomato: 0.5, Chicken: 2.0, Carrot: 0.4 };
  const optimized = inventory.map((item) => ({
    dish: `${item.name} Dish`,
    cost: (costs[item.name] || 1.0) * item.quantity,
  }));
  res.json(optimized.map((d) => `${d.dish} - Cost: $${d.cost.toFixed(2)}`));
});

app.get('/api/generate_dish', async (req, res) => {
  try {
    console.log('Fetching dish from Python...');
    const response = await axiosWithRetry('http://localhost:5001/api/generate_dish', 3, 1000);
    console.log('Python response:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('Generate error:', error.message, error.code, error.response?.data);
    res.status(500).json({ error: 'Failed to generate dish' });
  }
});

app.listen(5000, () => console.log('Server running on port 5000'));