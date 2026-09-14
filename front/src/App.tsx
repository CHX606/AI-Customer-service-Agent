import { App as AntApp, theme as antTheme } from "antd";
import zhCN from "antd/locale/zh_CN";
import { XProvider } from "@ant-design/x";
import xZhCN from "@ant-design/x/locale/zh_CN";
import { useEffect, useState } from "react";
import { CustomerServiceChat } from "./components/customer-service-chat";
import { useColorTheme } from "./hooks/use-color-theme";

export default function App() {
  const { theme, handleThemeToggle } = useColorTheme();
  const [reducedMotion, setReducedMotion] = useState(() => matchMedia("(prefers-reduced-motion: reduce)").matches);
  useEffect(() => {
    const media = matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReducedMotion(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  return (
    <XProvider locale={{ ...zhCN, ...xZhCN }} button={{ autoInsertSpace: false }} theme={{
      algorithm: theme === "dark" ? antTheme.darkAlgorithm : antTheme.defaultAlgorithm,
      token: {
        colorPrimary: theme === "dark" ? "#55b9a9" : "#168575",
        colorBgLayout: theme === "dark" ? "#10171e" : "#f5f7f8",
        colorBgContainer: theme === "dark" ? "#18222c" : "#ffffff",
        colorBgElevated: theme === "dark" ? "#1d2934" : "#ffffff",
        colorText: theme === "dark" ? "#dce4eb" : "#293740",
        colorTextSecondary: theme === "dark" ? "#a4b1bd" : "#657480",
        colorTextPlaceholder: theme === "dark" ? "#8899a8" : "#7a8994",
        fontFamily: '"Noto Sans SC Variable", "Microsoft YaHei UI", sans-serif',
        fontSize: 15, fontSizeSM: 13, fontWeightStrong: 600, lineHeight: 1.7,
        controlHeight: 40, borderRadius: 10, motion: !reducedMotion,
      },
      components: {
        Button: { fontWeight: 400 },
        Form: { itemMarginBottom: 20, verticalLabelPadding: "0 0 8px" },
        Tabs: { horizontalItemGutter: 24 },
        Modal: { titleFontSize: 20 },
      },
    }}>
      <AntApp className="customer-app">
        <CustomerServiceChat theme={theme} onThemeToggle={handleThemeToggle} />
      </AntApp>
    </XProvider>
  );
}
