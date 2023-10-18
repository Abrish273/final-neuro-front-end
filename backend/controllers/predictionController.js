// controllers/predictionController.js
const axios = require("axios");
const User = require("../models/User");
const StrokePrediction = require("../models/StrokePrediction");

// async function predictStrokeRisk(req, res) {
//   try {
//     const { data } = req.body; // Assuming you have the user's ID available in req.user
//     console.log("body", data);

//     // Forward the request to the Flask application
//     const flaskUrl = "http://127.0.0.1:4000"; // Change the port if necessary
//     const response = await axios.post(
//       `${flaskUrl}/predict_stroke_risk`,
//       data[0]
//     );

//     console.log("response", response);

//     // Store the prediction data in MongoDB
//     const predictionData = {
//       prediction: response.data["Logistic Regression Probability"],
//       data: data[0], // Save the input data used for the prediction
//     };

//     const prediction = new StrokePrediction(predictionData);
//     await prediction.save();

//     // Send the Flask application's response back to the client
//     res.json(response.data);
//   } catch (error) {
//     console.error(error);
//     res.status(500).json({ error: "Internal Server Error" });
//   }
// }
async function predictStrokeRisk(req, res) {
  try {
    // const { userId } = req.user; // Assuming you have the user's ID available in req.user
    // console.log("user",userId);
    // console.log("body",req.body);
    // Forward the request to the Flask application
    const data = req.body
    const flaskUrl = "http://127.0.0.1:4000"; // Change the port if necessary 
    const response = await axios.post(
      `${flaskUrl}/predict_stroke_risk`,
      data
    );

    // Store the prediction data in MongoDB
    // const predictionData = {
    //   // user: new mongoose.Types.ObjectId(userId), // Use 'new' to create a new ObjectId
    //   prediction: response.data["Logistic Regression Probability"],
    //   data: req.body.data[0], // Save the input data used for the prediction
    // };

    // const prediction = new StrokePrediction(predictionData);
    // await prediction.save();

    // Send the Flask application's response back to the client
    res.json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

async function getAllPredictions(req, res) {
  try {
    // Check if the user is authenticated and obtain the user's ID
    const userId = req.user.id; // Assuming you have a user object with ID after authentication

    // Fetch predictions for the logged-in user from the database
    const userPredictions = await StrokePrediction.find({ userId });

    // Send the user's predictions as a JSON response
    res.json({ predictions: userPredictions });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

async function getPredictionsByUserId(req, res) {
  try {
    // Extract the user ID from the request parameters
    const userId = req.params.userId;

    // Fetch predictions for the specified user from the database
    const userPredictions = await StrokePrediction.find({ userId });

    if (userPredictions.length === 0) {
      return res
        .status(404)
        .json({ error: "No predictions found for the specified user" });
    }

    // Send the user's predictions as a JSON response
    res.json({ predictions: userPredictions });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

async function getMedicalResponse(req, res) {
  try {
    const { question } = req.body; // Assuming the question is sent in the request body

    // Forward the question to the Flask application
    const flaskUrl = "http://127.0.0.1:4000"; // Change the port if necessary
    const response = await axios.post(`${flaskUrl}/medical`, { question });

    // Send the Flask application's response back to the client
    res.json({ response: response.data.response });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

// Function to call the '/get_stroke_recommendations' route in your Flask application
async function getStrokeRecommendationsFromFlask(data) {
  try {
    const flaskUrl = "http://127.0.0.1:4000"; // Change the URL and port if necessary

    // Make a POST request to the Flask route
    const response = await axios.post(
      `${flaskUrl}/get_stroke_recommendations`,
      data
    );

    // Return the response data
    return response.data;
  } catch (error) {
    console.error(error);
    throw new Error("Failed to call Flask route.");
  }
}

module.exports = {
  predictStrokeRisk,
  getAllPredictions,
  getMedicalResponse,
  getStrokeRecommendationsFromFlask,
  getPredictionsByUserId,
};
