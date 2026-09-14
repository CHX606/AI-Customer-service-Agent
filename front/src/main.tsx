import { createRoot } from "react-dom/client";
import App from "./App";
import "@fontsource-variable/noto-sans-sc";
import "antd/dist/reset.css";
import "./index.css";

createRoot(document.getElementById("root")!).render(
    <App />
);
