# pbh-python
Creating primordial black holes using python!

## Plotting column numbers

The column numbers are as given; the column names are in parentheses.

1. Grid point number (index)
2. \tilde{R} (r)
3. \tilde{U} (u)
4. \tilde{M} (m)
5. \tilde{\rho} (rho)
6. R (rfull)
7. U (ufull)
8. M (mfull)
9. \rho (rhofull)
10. 2M/R (horizon)
11. Characteristic speed c_s^+ (in \tilde{R}) (cs+)
12. Characteristic speed c_s^- (in \tilde{R}) (cs-)
13. Fluid speed c_s^0 (in \tilde{R}) (cs0)
14. \xi (xi)
15. Q (Q)
16. e^\phi (ephi)

## Helpful gnuplot commands

Data units are delimited by "\t", records by "\n", groups by "\n\n". Note the change from "\n\n\n". Time slices are separate data blocks rather than separate data sets, so they are addressed using `plot every` rather than `plot index`. This makes 3D plotting nicer. Also columns may be identified by name.

* General plotting of variables:
  ```gnuplot
  plot "output.dat" ev :::1::45 u 2:5 w l
  ```

Note that ev `ev` format is `:step::start::stop` (we don't use the omitted values).

* Plot characteristics in \tilde{R}:
  ```gnuplot
  plot "output.dat" ev 3 u 2:14:(column('cs+')*0.04):(0.01) w vec
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
