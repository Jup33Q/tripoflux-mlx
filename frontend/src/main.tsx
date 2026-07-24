// lkg.ts must be imported first: it installs the Looking Glass WebXR polyfill
// and deletes window.XRWebGLBinding before any WebGLRenderer is constructed.
import "./lkg";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
