/**
 * Registers what the unconstrained lab needs: the nine method modules of the `unconstrained`
 * family (TS ports of src/numopt/unconstrained/*.py) and the unconstrained test problems. The lab
 * and the dev catalog (#/dev) import this module; nothing else enters the lab's chunk.
 */
import '../../methods/unconstrained/first_order';
import '../../methods/unconstrained/newton';
import '../../methods/unconstrained/quasi_newton';
import '../../methods/unconstrained/conjugate_gradient';
import '../../methods/unconstrained/trust_region';
import '../../methods/unconstrained/derivative_free';
import '../../methods/unconstrained/accelerated';
import '../../methods/unconstrained/anderson';
import '../../methods/unconstrained/regularized_newton';
import '../../problems/unconstrained';

export {};
