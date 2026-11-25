# How to read the results of experiment 33

**TLDR:** There is no data split, no judge and no method comparison here. Each output line states whether an algebraic identity or inequality held on every random instance. Read errors as "should be near zero" and slacks as "should be nonnegative".

1. **Which data:** none. Rating distributions on K = 6 categories (ratings 0 to 5) are drawn from a symmetric Dirichlet(0.7) with a floor of 1e-9, seed 20261125. There is no train, validation or held-out test set, no human rating and no model call. All sums are exact finite sums over the K categories.
2. **Which metric:** per line, one of three quantities. (a) A worst absolute error between the two sides of an identity: it should be near zero; 1e-14 to 1e-13 is floating-point rounding and about 1e-9 is the finite-difference step used for gradients. (b) A smallest slack, right side minus left side of an inequality: it should be nonnegative, and a larger value means a looser bound. (c) A largest ratio of the left to the right side of a bound: it should be at most 1, and a smaller value means a looser bound. Reference: the exact Vandermonde determinant is the integer product 1! 2! 3! 4! 5! = 34,560.
3. **What is subtracted:** left side minus right side of one statement, on the same random instances. No two methods are subtracted from each other.
4. **Uncertainty:** none; the script is deterministic given the seed. The instance count of every check is printed in its line and listed in the README.
5. **How the judge was chosen:** no judge is chosen or fitted. The "CE selector" and "mean-only selector" in the last two lines are two selection rules compared by exact rational arithmetic, not trained models.
