import User from "../models/User.js";
import { NextFunction,Response,Request } from "express";
import { createToken } from "../utils/token-manager.js";
import { COOKIE_NAME } from "../utils/constants.js";
import { hash, compare } from "bcrypt";
export const getAllUsers = async (
    req: Request,
    res: Response,
    next: NextFunction,
  ) => {
    try {
      //get all users
      const users = await User.find();
      return res.status(200).json({ message: "OK", users });
    } catch (error) {
      console.log(error);
      return res.status(200).json({ message: "ERROR", cause: error.message });
    }
  };

export const userSignup=async(req: Request,
  res: Response,
  next: NextFunction,)=>{
    try {
      const users=await User.find();
      const { name, email, password } = req.body;
      const existingUser = await User.findOne({ email });
      if (existingUser) return res.status(401).send("User already registered");
      const hashedPassword = await hash(password, 10);
      const user = new User({ name, email, password: hashedPassword });
      await user.save();

       // create token and store cookie
    res.clearCookie(COOKIE_NAME, {
      httpOnly: true,
      domain: "localhost",
      signed: true,
      path: "/",
    });

    const token = createToken(user._id.toString(), user.email, "2d");
    const expires = new Date();
    expires.setDate(expires.getDate() + 2);
    res.cookie(COOKIE_NAME, token, {
      path: "/",
      domain: "localhost",
      expires,
      httpOnly: true,
      signed: true,
    });









      return res
      .status(201)
      .json({ message: "OK", name: user.name, email: user.email });
    } catch (error) {
      console.log(error);
      return res.status(200).json({ message: "ERROR", cause: error.message });
    }
  };


  export const userLogin = async (
    req: Request,
    res: Response,
    next: NextFunction
  ) => {
    try {
      //user login compare fun use for comparing passwords 
      const { email, password } = req.body;
      const user = await User.findOne({ email });
      if (!user) {
        return res.status(401).send("User not registered");
      }
      const isPasswordCorrect = await compare(password, user.password);
      if (!isPasswordCorrect) {
        return res.status(403).send("Incorrect Password");
      }
  
      // create token and store cookie
  
      res.clearCookie(COOKIE_NAME, {
        httpOnly: true,
        domain: "localhost",
        signed: true,
        path: "/",
      });
  
      const token = createToken(user._id.toString(), user.email, "2d");
      const expires = new Date();
      expires.setDate(expires.getDate() + 2);
      res.cookie(COOKIE_NAME, token, {
        path: "/",
        domain: "localhost",
        expires,
        httpOnly: true,
        signed: true,
      });
  
      return res
        .status(200)
        .json({ message: "OK", name: user.name, email: user.email });
    } catch (error) {
      console.log(error);
      return res.status(200).json({ message: "ERROR", cause: error.message });
    }
  };

  
  export const verifyUser = async (
    req: Request,
    res: Response,
    next: NextFunction
  ) => {
    try {
      //user token check
      const user = await User.findById(res.locals.jwtData.id);
      if (!user) {
        return res.status(401).send("User not registered OR Token malfunctioned");
      }
      if (user._id.toString() !== res.locals.jwtData.id) {
        return res.status(401).send("Permissions didn't match");
      }
      return res
        .status(200)
        .json({ message: "OK", name: user.name, email: user.email });
    } catch (error) {
      console.log(error);
      return res.status(200).json({ message: "ERROR", cause: error.message });
    }
  };
  
  //logout user
  export const userLogout = async (
    req: Request,
    res: Response,
    next: NextFunction
  ) => {
    try {
      //user token check
      const user = await User.findById(res.locals.jwtData.id);
      if (!user) {
        return res.status(401).send("User not registered OR Token malfunctioned");
      }
      if (user._id.toString() !== res.locals.jwtData.id) {
        return res.status(401).send("Permissions didn't match");
      }
  
      res.clearCookie(COOKIE_NAME, {
        httpOnly: true,
        domain: "localhost",
        signed: true,
        path: "/",
      });
  
      return res
        .status(200)
        .json({ message: "OK", name: user.name, email: user.email });
    } catch (error) {
      console.log(error);
      return res.status(200).json({ message: "ERROR", cause: error.message });
    }
  };