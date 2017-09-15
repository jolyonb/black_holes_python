# pbh-python
Creating primordial black holes using python

Here are a bunch of gnuplot commands that are helpful:

* General plotting of variables:
> plot "output.dat" i 1:50:1 u 1:(2*$7/$8) w l

* Plot characteristics in \tilde{R}:
> plot "output.dat" i 0:45:1 ev 3 u 2:14:($11*0.04):(0.01) w vec

* 3D Plot Misner-Sharp and Russel-Bloomfield evolutions on top of each other:
> splot 'outputrb.dat' i 400:600 u 2:14:4 w l, 'outputms.dat' i 400:600 u 2:14:4 w l
