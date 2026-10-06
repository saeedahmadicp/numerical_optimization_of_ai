/**
 * Error boundary around the lab stage: one bad frame shows a quiet notice with a retry button
 * instead of unmounting the whole app. `resetKey` (the runs' problem and methods) clears the
 * error when the input changes.
 */
import { Component, type ReactNode } from 'react';
import { Button } from '../../ui/components';
import styles from './CombinatorialLab.module.css';

interface Props {
  resetKey: string;
  children: ReactNode;
}
interface State {
  error: Error | null;
  key: string;
}

export class StageBoundary extends Component<Props, State> {
  override state: State = { error: null, key: this.props.resetKey };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    return props.resetKey !== state.key ? { error: null, key: props.resetKey } : null;
  }

  override componentDidCatch(error: Error) {
    console.error('Combinatorial stage:', error);
  }

  override render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className={styles.stageError} role="alert">
        <p>This frame could not be drawn: {this.state.error.message}.</p>
        <Button
          size="sm"
          variant="ghost"
          icon="reset"
          onClick={() => this.setState({ error: null })}
        >
          Draw again
        </Button>
      </div>
    );
  }
}
