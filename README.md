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

Data units are delimited by "\t", records by "\n", groups by "\n\n". Note the change from "\n\n\n". Time slices are separate data blocks rather than separate data sets, so they are addressed using `plot every` rather than `plot index`. This makes 3D plotting nicer. Also columns may be identified by name.

* General plotting of variables:
  ```gnuplot
  plot "output.dat" ev :::1::45 u "r":(2*column('M')/column('R')) w l
  ```

* Plot characteristics in \tilde{R}:
  ```gnuplot
  plot "output.dat" ev 3 u 2:14:(column('csp')*0.04):(0.01) w vec
  ```

* 3D Plot Misner-Sharp and Russel-Bloomfield evolutions on top of each other:
  ```gnuplot
  splot 'outputrb.dat' u 2:14:4 w l, 'outputms.dat' u 2:14:4 w l
  ```

* Countour Plotting for apparent horizon:
  ```gnuplot
  set contour
  set cntrparam levels discrete 0
  set view map
  splot "output.dat" u 2:14:"csp" w l
  ```

* To remove coordinate grid on horizon plot:
  ```gnuplot
  unset surf
  ```
