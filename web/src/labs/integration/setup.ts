/**
 * Registers what the quadrature lab needs: the integration port and the calculus problems.
 * Only these two modules are imported (not the global method glob), so the lab chunk carries
 * just its own family.
 */
import '../../methods/integration/methods';
import '../../problems/calculus';

export {};
