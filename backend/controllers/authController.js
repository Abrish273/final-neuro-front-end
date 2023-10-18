const User = require("../models/User");
const { StatusCodes } = require("http-status-codes");
const CustomError = require("../errors");
const { attachCookiesToResponse, createTokenUser } = require("../utils");
const jwt = require("jsonwebtoken");
require("dotenv").config();

const register = async (req, res) => {
  const data = req.body;
  // const { email, password } = data;
  // console.log(data);
  const { first_name,
    last_name,
    date_of_birth,
    age,
    gender,
    phone_number,
    address,
    city,
    country,
    email,
    password } =data
  const role = "user";
  // const Creator =email;

  // const encreptedPassword = await bcrypt.hash(password, 10);

  console.log(first_name,
    last_name,
    date_of_birth,
    age,
    gender,
    phone_number,
    address,
    city,
    country,
    email,
    password);

  try {
    await User.create({
      first_name,
    last_name,
    date_of_birth,
    age,
    gender,
    phone_number,
    address,
    city,
    country,
    email,
    password
    });

    console.log("success");

    res.send({ status: 200 });
  } catch (error) {
    if (error.code === 11000 || error.code === 16460) {
      // Duplicate key error or unique key constraint violation
      res.send({ status: "error", error: "Duplicate data" });
      console.log("Duplicate data");
    } else if (error.code === 17140) {
      // Missing expected field error
      res.send({ status: "error", error: "Missing expected field" });
      console.log("Missing expected field");
    } else if (error.code === 20250) {
      // Invalid document or field name error
      res.send({ status: "error", error: "Invalid document or field name" });
      console.log("Invalid document or field name");
    } else if (error.code === 21328) {
      // Maximum index key length exceeded error
      res.send({ status: "error", error: "Maximum index key length exceeded" });
      console.log("Maximum index key length exceeded");
    } else {
      // Generic error handling
      res.send({ status: "error", error: error.message });
      console.log(error.message);
    }
  }
};
// const register = async (req, res) => {
//   const {
//     data, // Add country to the request body
//   } = req.body;
//   const newnew =req.body;
//   // console.log(req.body);
//   // const newdata = JSON.parse(data);
//   console.log(newnew.email,
//     newnew.password,
//     newnew.first_name,
//     newnew.last_name,
//     newnew.date_of_birth,
//     newnew.phone_number,
//     newnew.address,
//     newnew.city,
//     newnew.age,
//     newnew.gender,
//     newnew.country,);
//   try {
//     // Check if email already exists
//     const emailAlreadyExists = await User.findOne({ email });
//     if (emailAlreadyExists) {
//       throw new CustomError.BadRequestError("Email already exists");
//     }

//     // Create a new user with the provided fields
//     const newUser = await User.create({
//       email:newnew.email,
//       password:newnew.password,
//       first_name:newnew.first_name,
//       last_name:newnew.last_name,
//       date_of_birth:newnew.date_of_birth,
//       phone_number:newnew.phone_number,
//       address:newnew.address,
//       city:newnew.city,
//       age:newnew.age, // Include age
//       gender:newnew.gender, // Include gender
//       country:newnew.country, // Include country
//     });

//     // Set the default role to "user"
//     const role = "user";

//     // Additional validation can be added here if necessary

//     const userWithToken = createTokenUser(newUser);
//     attachCookiesToResponse({ res, user: userWithToken });

//     res.status(StatusCodes.CREATED).json({ user: userWithToken });
//   } catch (error) {
//     // Handle validation errors or other errors and send an appropriate response
//     if (error.name === "ValidationError") {
//       const errors = Object.values(error.errors).map((err) => err.message);
//       res.status(StatusCodes.BAD_REQUEST).json({ error: errors });
//     } else {
//       // Handle other types of errors
//       res
//         .status(StatusCodes.INTERNAL_SERVER_ERROR)
//         .json({ error: error.message });
//     }
//   }
// };

const login = async (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    throw new CustomError.BadRequestError("Please provide email and password");
  }

  const user = await User.findOne({ email });

  if (!user) {
    throw new CustomError.UnauthenticatedError("Invalid Credentials");
  }

  const isPasswordCorrect = await user.comparePassword(password);

  if (!isPasswordCorrect) {
    throw new CustomError.UnauthenticatedError("Invalid Credentials");
  }

  // Use the secret key and token expiration from environment variables
  const secretKey = process.env.JWT_SECRET;
  const tokenExpiration = process.env.JWT_LIFETIME;

  if (!secretKey) {
    throw new CustomError.InternalServerError(
      "JWT secret key is not configured."
    );
  }

  if (!tokenExpiration) {
    throw new CustomError.InternalServerError(
      "Token expiration is not configured."
    );
  }

  // Define the payload for the JWT
  const payload = {
    userId: user._id,
    email: user.email,
    firstName: user.first_name,
    lastName: user.last_name,
    role: user.role,
  };

  // Generate a JSON Web Token (JWT) with the configured expiration time
  const token = jwt.sign(payload, secretKey, { expiresIn: tokenExpiration });

  res.status(StatusCodes.OK).json({
    user: {
      _id: user._id,
      email: user.email,
      token: token,
      firstName: user.first_name,
      lastName: user.last_name,
      role: user.role,
    },
  });
};

const logout = async (req, res) => {
  res.cookie("token", "logout", {
    httpOnly: true,
    expires: new Date(Date.now() + 1000),
  });
  res.status(StatusCodes.OK).json({ msg: "user logged out!" });
};

module.exports = {
  register,
  login,
  logout,
};
