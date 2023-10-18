const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const validator = require("validator");

// Define the user schema
const UserSchema = new mongoose.Schema({
  first_name: {
    type: String,
    required: [true, "Please provide first name"],
    minlength: 2,
    maxlength: 50,
  },
  last_name: {
    type: String,
    required: [true, "Please provide last name"],
    minlength: 2,
    maxlength: 50,
  },
  date_of_birth: {
    type: Date,
    required: [true, "Please provide date of birth"],
  },
  age: {
    type: Number,
    required: [true, "Please provide age"],
    min: 18,
  },
  gender: {
    type: String,
    enum: ["male", "female", "other"],
    required: [true, "Please provide gender"],
  },
  phone_number: {
    type: String,
    required: [true, "Please provide phone number"],
    // validate: {
    //   validator: validator.isMobilePhone,
    //   message: "Please provide a valid phone number",
    // },
  },
  address: {
    type: String,
    required: [true, "Please provide address"],
    minlength: 5,
    maxlength: 100,
  },
  city: {
    type: String,
    required: [true, "Please provide city"],
    minlength: 2,
    maxlength: 50,
  },
  country: {
    type: String,
    required: [true, "Please provide country"],
  },
  email: {
    type: String,
    unique: true,
    required: [true, "Please provide email"],
    validate: {
      validator: validator.isEmail,
      message: "Please provide a valid email",
    },
  },
  password: {
    type: String,
    required: [true, "Please provide password"],
    minlength: 6,
  },
  role: {
    type: String,
    enum: ["user"],
    default: "user",
  },
});

// Hash the password before saving
UserSchema.pre("save", async function (next) {
  if (!this.isModified("password")) return next();
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

// Add the 'comparePassword' method to compare passwords
UserSchema.methods.comparePassword = async function (candidatePassword) {
  const isMatch = await bcrypt.compare(candidatePassword, this.password);
  return isMatch;
};

// Create and export the User model
const User = mongoose.model("User", UserSchema);

module.exports = User;
