# Black Hole Evolution Code

This is the clean implementation of the black hole evolution code. Apart from being nicely structured and readable, it also links all physics equations in the code to equations in [this version of the paper](paper.pdf). Sometimes we have two versions of an equation coded, with one commented out. The reason for this is to reuse things that have already been computed, so as to cut down on computation. Note that we specialize to w=1/3 and alpha=1/2 (alpha is just a function of w). Also note that we take the only scale in the problem R_H=1. All dimensionful quantities can be reconstructed by reinserting factors of R_H (the horizon radius at the start of evolution). Unfortunately, I haven't finished updating the PDF file completely; there are a few issues towards the end of the file where I haven't yet propagated some redefinitions. Such issues are noted in the code where applicable.

The entry point for the code is driver.py, which uses an old algorithm to take a linearized \delta_m and construct the growing mode from it. (We have better tools now.)

I recommend using gnuplot to visualize the output (the output has been formatted according to gnuplot specifications). Some helpful plotting commands are listed in the readme file for this repository.


## Lagrangian Evolution

The Lagrangian evolution is characterized by having grid points move along with fluid elements. This means that the grid is continually changing. This is fine, except that when you have a large overdensity, the grid tends to fall towards that overdensity. This means that you tend to have a lot of grid points near the origin, then some very sparsely distributed points, then evenly distributed points once you get back into the cosmological regime. This is fine, until you want a high accuracy derivative in that sparsely distributed area. If a shock wave passes through this area, you're going to be in trouble, as the derivative quality is terrible there, and will lead to instabilities.

To diagnose this, try the following plots:

* Plot of grid point index as a function of radius
```
plot "output_lag.dat" ev :::2:: u 2:1 w p
```

* Plot of Q as a function of radius (this is where bad derivatives are really felt)
```
plot "output_lag.dat" ev :::2:: u 2:15 w p
```


## Eulerian Evolution

In the Eulerian evolution, the grid points stay at fixed radius. What this tends to mean is that you need to run a non-uniform grid so that you have resolution where you need it. If you're trying to resolve shocks, you may just need a lot of grid points everywhere!

Note that if you're getting integration errors from shocks, increasing resolution will usually help.

I'm presently having trouble with:
(gridpoints=500, squeeze=2, Amax=10, amplitude=0.1737, sigma=2.0)
I'm wondering if the old code can handle this situation? If so, then the only real difference between the new code and the old is the 4th order derivatives for computing Q.

Old code barfed with this too. Trying again with 1000 gridpoints. Also barfed. Note - this was using viscosity=20, so a lot of suppression. One last go at 1500 gridpoints. Also failed. In particular, failed before a shock wave formed. Looked like a high frequency instability in rho was responsible (in a position without a huge amount of nonlinearity), suggesting that we had complex eigenvalues in our differentiation matrix? May need to test this carefully... Also just need to test this in our new code.

Upshot: I need to clean up the code in a few more places so we can run things faster, but we're seeing the same issues as previously :( Hopefully with cleaner code in place, we can investigate more thoroughly, and potentially with more eyes on it too.
