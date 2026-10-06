/**
 * Registers what the roots lab needs: the scalar root finders (bracketing + open) and the roots
 * problems. Only these modules — the lab chunk does not pull in other families.
 */
import '../../methods/roots/bracketing';
import '../../methods/roots/open';
import '../../problems/roots';

export {};
