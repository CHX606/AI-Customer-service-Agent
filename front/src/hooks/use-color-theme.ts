import { useState } from "react";

const THEME_STORAGE_KEY = "customer-service-theme";

type ColorTheme = "light" | "dark";

function getInitialTheme(): ColorTheme {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}


export function useColorTheme() {
  const [theme, setTheme] = useState<ColorTheme>(getInitialTheme);
  const handleThemeToggle = () => {
    setTheme((current) => {
      const nextTheme = current === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = nextTheme;
      document.documentElement.style.colorScheme = nextTheme;
      localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
      return nextTheme;
    });
  };


  return { theme, handleThemeToggle };
}
