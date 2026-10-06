/**
 * Registers what the systems lab needs: the Newton/Broyden port and the systems problems. Only
 * these two modules are imported (not the global globs), so the lab chunk stays small.
 */
import '../../methods/roots/systems';
import '../../problems/systems';

export {};
