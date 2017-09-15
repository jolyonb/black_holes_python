# pbh-python
Creating primordial black holes using python!

## Plotting column numbers

1. Grid point number
2. \tilde{R}
3. \tilde{U}
4. \tilde{M}
5. \tilde{\rho}
6. R
7. U
8. M
9. \rho
10. 2M/R
11. Characteristic speed c_s^+ (in \tilde{R})
12. Characteristic speed c_s^- (in \tilde{R})
13. Characteristic speed c_s^0 (in \tilde{R})
14. \xi

## Helpful gnuplot commands

* General plotting of variables:
> plot "output.dat" i 1:50:1 u 1:(2*$7/$8) w l

* Plot characteristics in \tilde{R}:
> plot "output.dat" i 0:45:1 ev 3 u 2:14:($11*0.04):(0.01) w vec

* 3D Plot Misner-Sharp and Russel-Bloomfield evolutions on top of each other:
> splot 'outputrb.dat' i 400:600 u 2:14:4 w l, 'outputms.dat' i 400:600 u 2:14:4 w l

* Countour Plotting for Aparrent horizon:
```gnuplot
set contour
set cntrparam levels discrete 0
set view map
splot "< sed '/^#/ d' output.dat | cut -f2,11,14 | uniq" u 1:3:2 w l
```

* To remove coordinate grid on horizon plot:
```gnuplot
unset surf
```

