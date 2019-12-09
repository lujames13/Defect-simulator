# Defect simulator

## Table of content
  - [Motivation](#motivation)
  - [Ideas](#ideas)
  - [To-Do list](#to-do-list)
  - [Reference](#reference)

## Motivation
In condensed matter system, system will progress to the structure which's total free energy is the lowest.  
Now giving the free energy function with constant k, b
![equation](https://latex.codecogs.com/gif.latex?F&space;=&space;\int&space;\int&space;\left&space;[&space;\frac{b}{2}\left&space;(&space;\nabla^{2}\rho&space;&plus;&space;k^{2}&space;\right&space;)^{2}-\frac{r}{2}\rho^{2}&space;&plus;&space;\frac{1}{4}\rho^{4}&space;\right&space;]dxdy)  
and find out the density function ![rho](https://latex.codecogs.com/gif.latex?\rho(x,&space;y)) that produce the lowest F.  
If we put the system in the square box with length L, the density function will develop to the periodic form. But if we take the first reesult and extrude the box size to (0.5L, 2L), there are some defect will be produce.  
If we can simulate the defect production with custum constant k, b and the different extrusion, the related research will be benefit.
## Ideas
### Targets of Project
 - Simulate and virtualize the simulation of defect production.
 - Let user adjust the constant k,b and the extrusion.
 - Take the sanpshot of the precess, and let user can watch the defect producing process foward or backward freely.
### Project Structure
The calculation part is weitten in `C++` for the speed, and the visualization part is writtenin `Python` for the accessbility

### To-Do list
 - [ ] define the data structure
 - [ ] 1D version
 - [ ] 2d version
### Reference
 - [Defect formation in the Swift-Hohenberg equation](http://estebanmoro.org/pdf/Defect_formation_in_the_Swift_Hohenberg_equation_.pdf)
 - [Swift–Hohenberg equation](https://en.wikipedia.org/wiki/Swift%E2%80%93Hohenberg_equation)
 - [Kibble–Zurek mechanism](https://en.wikipedia.org/wiki/Kibble%E2%80%93Zurek_mechanism)
 - [Rayleigh–Bénard convection](https://en.wikipedia.org/wiki/Rayleigh%E2%80%93B%C3%A9nard_convection)
