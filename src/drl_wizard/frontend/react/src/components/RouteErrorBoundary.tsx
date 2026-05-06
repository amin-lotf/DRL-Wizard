import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface RouteErrorBoundaryProps {
  routeKey: string;
  children: ReactNode;
}

interface RouteErrorBoundaryState {
  hasError: boolean;
  message: string;
}

export class RouteErrorBoundary extends Component<
  RouteErrorBoundaryProps,
  RouteErrorBoundaryState
> {
  state: RouteErrorBoundaryState = {
    hasError: false,
    message: "",
  };

  static getDerivedStateFromError(error: Error): RouteErrorBoundaryState {
    return {
      hasError: true,
      message: error.message || "The page failed to render.",
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Route render failed", error, errorInfo);
  }

  componentDidUpdate(prevProps: RouteErrorBoundaryProps) {
    if (prevProps.routeKey !== this.props.routeKey && this.state.hasError) {
      this.setState({ hasError: false, message: "" });
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="panel p-6">
          <div className="eyebrow">Route error</div>
          <h2 className="mt-2 text-xl font-semibold text-text">This page crashed</h2>
          <p className="mt-2 max-w-2xl text-sm text-muted">
            {this.state.message || "An unexpected client-side error occurred while switching pages."}
          </p>
          <button
            className="btn-primary mt-5"
            type="button"
            onClick={() => window.location.reload()}
          >
            Reload application
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
