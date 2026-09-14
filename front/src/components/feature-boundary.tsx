import { Component, type ErrorInfo, type ReactNode } from "react";
import { Button, Result } from "antd";

type Props = { children: ReactNode; name: string; onClose?: () => void };

/** Keep a feature rendering failure isolated; hooks handle request failures. */
export class FeatureBoundary extends Component<Props, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`${this.props.name}渲染失败`, error, info.componentStack);
  }
  render() {
    if (!this.state.failed) return this.props.children;
    return <Result status="error" title={`${this.props.name}暂时无法显示`} subTitle="请重试，或关闭后重新打开。" extra={[
      <Button key="retry" type="primary" onClick={() => this.setState({ failed: false })}>重试</Button>,
      this.props.onClose && <Button key="close" onClick={this.props.onClose}>关闭</Button>,
    ]} />;
  }
}
