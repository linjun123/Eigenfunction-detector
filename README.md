# Eigenfunction-detector




###1. 
The eigenfunctions of a Schrodinger equation are important objects in physics and mathematics. The eigenfunction finding algorithms are knowning to be costy when the sample size is large. On the other hand, solving a Poisson equation with potential can be reduced to solve a linear equation which is easier than the eigenfunction problem. 
The landscape theory made it possible to find eigenfunctions by solving a Poisson equation.

More precisely, consider the eigenvalue problem of Schrodinger equation 
$$
  \Delta u + V u = \lambda u
$$
in region $\Omega$.
In order to find the solution $u$, we can solve a discretized problem and find the eigenvalues and eigenvectors of matrix $\Delta + V$.

In certain situations, we only need the first a few eigenfunctions and we somehow know that the eigenfunctions are localized in space. In these cases, the landscape provides a fast and efficient way for us to find these eigenfunctions' places.

The landscape function $g$ is the solution of the following Poisson equation
$$
 \Delta g + V g = 1
$$
where $1$ denote the constant function in the region.
Under certain conditions,
the eigenfunctions are localized at the places where the landscape function $g$ tends to be large. 

We consider here the simple case when the region $\Omega$ is the two dimensional box with edge length $n$.
###2.
There are two challenges:

(a) how to read from the landscape function to decide the places where the eigenfunctions are seems to depend on the researcher. 

(b) even if one can determine the first several eigenfunctions' locations, it seems uncertain to decide which eigenfunction localizes at which place where $g$ is large.

Especially, the second question is hard because sometimes the first a few eigenvalues are close to each other.

###3.

The current implementation focuses on one dimensional case and the values of the first a few eigenvalues. See https://github.com/nehcili/Wave-Localization

###4.

We first calculate the landscape function which reduces to solve a linear equation. Then we 'read' from the landscape function to decide the places of first a few eigenfunctions.

The 'read' part follows the previous 1d work by training a neural network. More precisely, we are facing a localization problem. The neural net needs to:

(a) find the places of the eigenfunctions

(b) label them from the ordering of the corresponding eigenvalues

###5.

An "automated eigenfunctions reader" which is different from the previous method of solving an eigenvalue problem. 


Link for the training data: 
30000 potentials on 60*60 grid:
https://drive.google.com/file/d/1--SqecqSHLE_UIw2cEcVuyu2koryqcD7/view?usp=sharing

30000 corresponding Landscape functions:
https://drive.google.com/file/d/1-2pfp5niZgoGp9qo8XhR97_lV7duRTjg/view?usp=sharing

30000 corresponding targets (eigenvalues, eigenfunctions' standard deviations and locations):
https://drive.google.com/file/d/1Xm6ma_IPqcnsS9rmoEpesFoFTLk_G7wp/view?usp=sharing

