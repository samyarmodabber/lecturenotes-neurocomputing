# Math basics

This chapter is not part of the course itself (there will not be questions at the exam on basic mathematics) but serves as a reminder of the important mathematical notions that are needed to understand this course. Students who have studied mathematics as a major can safely skip this part, as there is nothing fancy (although the section on information theory could be worth a read). 

It is not supposed to replace any course in mathematics (we won't show any proof and will skip what we do not need) but rather to provide a high-level understanding of the most important concepts and set the notations. Nothing should be really new to you, but it may be useful to have everything summarized at the same place.

**References:** Part I of Goodfellow et al. (2016) {cite}`Goodfellow2016`. Any mathematics textbook can be used in addition.

## Linear algebra

Several mathematical objects are manipulated in linear algebra:

* **Scalars** $x$ are 0-dimensional values (single numbers, so to speak). They can either take real values ($x \in \Re$, e.g. $x = 1.4573$, floats in CS) or natural values ($x \in \mathbb{N}$, e.g. $x = 3$, integers in CS). 

* **Vectors** $\mathbf{x}$ are 1-dimensional arrays of length $d$. The bold notation $\mathbf{x}$ will be used in this course, but you may also be accustomed to the arrow notation $\overrightarrow{x}$ used on the blackboard. Vectors are typically represented vertically to outline its $d$ elements $x_1, x_2, \ldots, x_d$:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix}$$

When using real numbers, the **vector space** with $d$ dimensions is noted $\Re^d$, so we can note $\mathbf{x} \in \Re^d$. 

* **Matrices** $A$ are 2-dimensional arrays of size (or shape) $m \times n$ ($m$ rows, $n$ columns). They are represented by a capital letter to distinguished them from scalars. The element $A_{i,j}$ of a matrix $A$ is the value on the $i$-th row and $j$-th column. For example with a $3 \times 3$ matrix::

$$A = \begin{bmatrix}
A_{1, 1} & A_{1, 2} & A_{1, 3} \\
A_{2, 1} & A_{2, 2} & A_{2, 3} \\
A_{3, 1} & A_{3, 2} & A_{3, 3} \\
\end{bmatrix}$$


## Calculus

## Probability theory

## Statistics

## Information theory

<https://towardsdatascience.com/entropy-cross-entropy-and-kl-divergence-explained-b09cdae917a>