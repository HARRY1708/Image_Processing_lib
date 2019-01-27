set terminal png  size 5.0,3.0
set output "speed_Plot.png"
set format y "$%g$"
set format x '$%5.1f\mu$'
set title "SpeedPlot"
set label "Data" at -5,-5 right
set arrow from -5,-5 to -3.3,-6.7
set key top left
set xtic -10,5,10
load "pthread.dat" with errorbars
load "mkl.dat" with errorbars
load "openblas.dat" with errorbars
load "naive.dat" with errorbars