
// routes/predictionRoutes.js
const express = require('express');
const router = express.Router();
const predictionController = require('../controllers/predictionController');
const {
  authenticateUser,
 
} = require("../middleware/authentication");

// Define the route for predicting stroke risk
router.post('/predict_stroke_risk', predictionController.predictStrokeRisk);
router.get("/predictions",authenticateUser, predictionController.getAllPredictions);
router.post("/medical", predictionController.getMedicalResponse);
router.post("/getStrokeRecommendations", async (req, res) => {
  try {
    const data = req.body; // You can pass the necessary data in the request body
    const response =
      await predictionController.getStrokeRecommendationsFromFlask(data);
    res.json(response);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});


module.exports = router;
