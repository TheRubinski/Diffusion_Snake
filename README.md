# Diffusion Snake, a white box edge detector
2 People ~ 3.5 weeks

Using [Creemers (2002)](https://cvg.cit.tum.de/_media/spezial/bib/cremers_dissertation.pdf)

A white-box segmentation approach using gradient descent on a Mumford-Sha functional. Use of an arbitrarily initialized spline curve.
There is a simplified and a full version of the functional. We did implement both. See the demo files. We have also added a respacing for the spline control points to speed up convergence.
The hardest part was getting through the math, as the paper is quite old and no code nor pseudo code, and the explanation of the formulars is quite sketchy in places.

https://github.com/TheRubinski/Diffusion_Snake/assets/74351447/7268e7fa-195b-4dc5-baf4-dc9a2cd859bb
