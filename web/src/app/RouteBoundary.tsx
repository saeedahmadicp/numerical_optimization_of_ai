import { Component, type ErrorInfo, type ReactNode } from 'react';
import type { Crumb } from './AppHeader';
import { PageError } from './Pages';

interface Props {
  title: string;
  section?: Crumb;
  children: ReactNode;
}
interface State {
  error: Error | null;
  /** Bumped by "Try again": re-mounts the page from scratch. */
  attempt: number;
}

/**
 * Catches a render error in one route (a lab, a method page, a study) and shows a recoverable
 * error panel instead of a blank page. App keys it by route, so navigating away resets it.
 */
export class RouteBoundary extends Component<Props, State> {
  override state: State = { error: null, attempt: 0 };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  override componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error('Route render error', error, info.componentStack);
  }

  private retry = () => this.setState((s) => ({ error: null, attempt: s.attempt + 1 }));

  override render() {
    const { error, attempt } = this.state;
    if (error)
      return (
        <PageError
          title={this.props.title}
          section={this.props.section}
          message={error.message}
          onRetry={this.retry}
        />
      );
    return (
      <div key={attempt} style={{ display: 'contents' }}>
        {this.props.children}
      </div>
    );
  }
}
