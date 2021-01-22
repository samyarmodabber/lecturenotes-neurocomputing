# Reservoir computing

Slides: [pdf](https://www.tu-chemnitz.de/informatik/KI/edu/neurocomputing/lectures/pdf/4.5-Reservoir.pdf)


The concept of **Reservoir Computing** (RC) was developed simultaneously by two researchers at the beginning of the 2000s.
RC builds on the idea of Hopfield networks but focuses on the dynamics rather than on the fixed points.
Herbert Jaeger (Uni Bremen) introduced **echo-state networks** (ESN) using rate-coded neurons {cite}`Jaeger2001`.
Wolfgang Maass (TU Graz) introduced **liquid state machines** (LSM) using spiking neurons {cite}`Maass2002`.


## Echo-state networks (ESN)

### Structure

An ESN is a set of **recurrent** units (sparsely connected) exhibiting complex spatiotemporal dynamics.

```{figure} ../img/RC-principle2.png
---
width: 100%
---
Structure of an echo-state network. Source: {cite}`Tanaka2019a`.
```

Rate-coded neurons in the reservoir integrate inputs and recurrent connections using an ODE:

$$
    \tau \, \frac{dx_j(t)}{dt} + x_j(t) = \sum_i W^\text{IN}_{ij} \, I_i(t) + \sum_i W_{ij} \, r_i(t) + \xi(t)
$$

The output of a neuron uses the tanh function (between -1 and 1):

$$
    r_j(t) = \tanh(x_j(t))
$$

**Readout neurons** (or output neurons) transform linearly the activity in the reservoir:

$$
    z_k(t) = \sum_j W^\text{OUT}_{jk} \, r_j(t) 
$$

In the original version of the ESN, only the readout weights are learned, not the recurrent ones.
One can use **supervised learning** to train the readout neurons to reproduce desired targets.

Inputs $\mathbf{I}(t)$ bring the **recurrent units** in a given state (like the bias in Hopfield networks).
The recurrent connections inside the reservoir create different **dynamics** $\mathbf{r}(t)$ depending on the strength of the weight matrix.
Readout neurons **linearly** transform the recurrent dynamics into temporal outputs $\mathbf{z}(t)$. 
**Supervised learning** (perceptron, LMS) trains the readout weights to reproduce a target $\mathbf{t}(t)$.
It is similar to a MLP with one hidden layer, but the hidden layer has dynamics.

Reservoirs only need a few hundreds of units in the reservoir to learn complex functions (e.g. $N=200$).
The recurrent weights are initialized randomly using a **normal distribution** with mean 0 and deviation $\frac{g}{\sqrt{N}}$:

$$w_{ij} \sim \mathcal{N}(0, \frac{g}{\sqrt{N}})$$

$g$ is a **scaling factor** characterizing the strength of the recurrent connections, what leads to different dynamics.

The recurrent weight matrix is often **sparse**:  A subset of the possible connections $N \times N$ has non-zero weights.
Typically, only 10% of the possible connections are created.

### Dynamics and edge of chaos

Depending on the value of $g$, the dynamics of the reservoir exhibit different attractors.
Let's have a look at the activity of a few neurons after the presentation of a short input.

When $g<1$, the network has no dynamics: the activity quickly fades to 0 when the input is removed.

![](../img/reservoir-dynamics-0.png)

For $g=1$, the reservoir exhibits some **transcient dynamics** but eventually fades to 0 (echo-state property).

![](../img/reservoir-dynamics-1.png)

For $g=1.5$, the reservoir exhibits many **stable attractors** due to its rich dynamics (Hopfield-like).

![](../img/reservoir-dynamics-2.png)

For higher values of $g$, there are no stable attractors anymore: **chaotic behavior**.

![](../img/reservoir-dynamics-3.png)

For $g = 1.5$, different inputs (initial states) lead to different attractors.

![](../img/reservoir-dynamics-represent.png)

The weight matrix must have a scaling factor above 1 to exhibit non-zero attractors.

```{figure} ../img/reservoir-dynamics-attractor.png
---
width: 60%
---
```

For a single input, the attractor is always the same, even in the presence of noise or perturbations. 

![](../img/reservoir-dynamics-reproduce1.png)

In the chaotic regime, the slightest uncertainty on the initial conditions (or the presence of noise) produces very different trajectories on the long-term.

![](../img/reservoir-dynamics-reproduce2.png)

The chaotic regime appears for $g > 1.5$. $g=1.5$ is the **edge of chaos**: the dynamics are very rich, but the network is not chaotic yet.

```{figure} ../img/reservoir-dynamics-chaos.png
---
width: 60%
---
```

```{admonition} Lorenz attractor

The **Lorenz attractor** is a famous example of a chaotic attractor.
The position $x, y, z$ of a particle is describe by a set of 3 **deterministic** ordinary differential equations:

$$\frac{dx}{dt} = \sigma \, (y -  x)$$
$$\frac{dy}{dt} = x \, (\rho - z) - y$$
$$\frac{dz}{dt} = x\, y - \beta \, z$$

The resulting trajectories over time have complex dynamics and are **chaotic**: the slightest change in the initial conditions generates different trajectories.

<div class='embed-container'><iframe src='https://www.youtube.com/embed/dP3qAq9RNLg' frameborder='0' allowfullscreen></iframe></div>

```

### Universal approximation

Using the reservoir as input, the linear readout neurons can be trained to reproduce **any non-linear** target signal over time:

$$
    z_k(t) = \sum_j W^\text{OUT}_{jk} \, r_j(t) 
$$

As it is a regression problem, the **delta learning rule** (LMS) is often enough.

$$
    \Delta W^\text{OUT}_{jk} = \eta \, (t_k(t) - z_k(t)) \, r_j(t) 
$$

```python
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(r, t)
```


```{figure} ../img/reservoir-fit.png
---
width: 70%
---
```

Reservoirs are **universal approximators**: given enough neurons in the reservoir and dynamics at the edge of the chaos, a RC network can approximate any non-linear function between an input signal $\mathbf{I}(t)$ and a target signal $\mathbf{t}(t)$.

The reservoir projects a low-dimensional input into a high-dimensional **spatio-temporal feature space** where trajectories becomes linearly separable. The reservoir increases the distance between the input patterns.
Input patterns are separated in both space (neurons) and time: the readout neurons need much less weights than the equivalent MLP: **better generalization and faster learning**.
The only drawback is that it does not deal very well with high-dimensional inputs (images).

```{figure} ../img/rc-patternseparation.png
---
width: 100%
---
Spatio-temporal pattern separation. Source: Seoane, L. F. (2019). Evolutionary aspects of reservoir computing. Philosophical Transactions of the Royal Society B. doi:10.1098/rstb.2018.0377.
```

The output of the readout neurons can be **fed back** into the reservoir to stabilize the trajectories:

$$
    \tau \, \frac{dx_j(t)}{dt} + x_j(t) = \sum_i W^\text{IN}_{ij} \, I_i(t) + \sum_i W_{ij} \, r_i(t) + \sum_i W^\text{FB}_{kj} \, z_k(t) + \xi(t)
$$

This makes the reservoir much more robust to perturbations, especially at the edge of chaos.
The trajectories are more stable (but still highly dynamical), making the job of the readout neurons easier.


### Applications

**Forecasting**: ESN are able to predict the future of chaotic systems (stock market, weather) much better than static NN.

```{figure} ../img/rc-forecasting.png
---
width: 100%
---
Forecasting. Source: <https://towardsdatascience.com/predicting-stock-prices-with-echo-state-networks-f910809d23d4>
```

**Physics:** RC networks can be used to predict the evolution of chaotic systems (Lorenz, Mackey-Glass, Kuramoto-Sivashinsky) at very long time scales (8 times the Lyapunov time).

```{figure} ../img/rc-flame.gif
---
width: 100%
---
Prediction of chaotic systems. Source: Pathak, J., Hunt, B., Girvan, M., Lu, Z., and Ott, E. (2018). Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach. Physical Review Letters 120, 024102–024102. doi:10.1103/PhysRevLett.120.024102.
```

**NLP:** RC networks can grasp the dynamics of language, i.e. its **grammar**.
RC networks can be trained to produce **predicates** ("hit(Mary, John)") from sentences ("Mary hit John" or "John was hit by Mary")

```{figure} ../img/rc-hinaut.png
---
width: 100%
---
Reservoirs for language understanding. Source: Hinaut, X., and Dominey, P. F. (2013). Real-Time Parallel Processing of Grammatical Structure in the Fronto-Striatal System: A Recurrent Network Simulation Study Using Reservoir Computing. PLOS ONE 8, e52946. doi:10.1371/journal.pone.0052946.
```

<div class='embed-container'><iframe src='https://www.youtube.com/embed/AUbJAupkU4M' frameborder='0' allowfullscreen></iframe></div>

The cool thing with reservoirs is that they do not have to be simulated by classical von Neumann architectures (CPU, GPU).
Anything able to exhibit dynamics at the edge of chaos can be used: VLSI (memristors), magnetronics, photonics (lasers), spintronics (nanoscale electronics)...  This can limit drastically the energy consumption of ML algorithms (200W for a GPU). Even biological or physical systems can be used...

```{figure} ../img/rc-memristor.jpg
---
width: 60%
---
Memristor-based RC networks. Source: {cite}`Tanaka2019a`.
```


```{figure} ../img/liquidbrain.png
---
width: 100%
---
A bucket of water can be used as a reservoir. Different motors provide inputs to the reservoir by creating weights. The surface of the bucket is recorded and used as an input to a linear algorithm. It can learn non-linear operations (XOR) or even speech recognition.. Source: Fernando, C., and Sojakka, S. (2003). Pattern Recognition in a Bucket. in Advances in Artificial Life Lecture Notes in Computer Science. doi:10.1007/978-3-540-39432-7_63.
```

Real biological neurons can be kept alive in a culture and stimulated /recorded to implement a reservoir.

```{figure} ../img/rc-culture2.jpg
---
width: 100%
---
Reservoir of biological neurons. Source: Frega, M., Tedesco, M., Massobrio, P., Pesce, M., and Martinoia, S. (2014). Network dynamics of 3D engineered neuronal cultures: a new experimental model for in-vitro electrophysiology. Scientific Reports 4, 1–14. doi:10.1038/srep05489.
```

Escherichia Coli bacteria change their mRNA in response to various external factors (temperature, chemical products, etc) and interact with each other.
Their mRNA encode a dynamical trajectory reflecting the inputs.
By placing them on a microarray, one can linearly learn to perform non-linear operations on the inputs.


```{figure} ../img/rc-ecoli.png
---
width: 70%
---
Reservoir of e-coli bacteria. Source: Jones, B., Stekel, D., Rowe, J., and Fernando, C. (2007). Is there a Liquid State Machine in the Bacterium Escherichia Coli? in 2007 IEEE Symposium on Artificial Life, 187–191. doi:10.1109/ALIFE.2007.367795.
```

ESN use the `tanh` activation function (between -1 and +1) and the weights can take any value.
In the brain, neurons are either excitatory (positive outgoing weights) or inhibitory (negative outgoing weights), never both (**Dale's law**).
Firing rates (outputs) are positive by definition.
It is possible to build ESN with a ratio 80% / 20% of excitatory and inhibitory cells, using ReLU transfer functions. A bit less stable, but works.

```{figure} ../img/EI.png
---
width: 100%
---
Excitatory-inhibitory reservoir. Source: Mastrogiuseppe, F., and Ostojic, S. (2017). Intrinsically-generated fluctuating activity in excitatory-inhibitory networks. PLOS Computational Biology 13, e1005498–e1005498. doi:10.1371/journal.pcbi.1005498.
```

RC networks can be used to model different areas, including the cerebellum, the olfactory system, the hippocampus, cortical columns, etc.
The brain has a highly dynamical recurrent architecture, so RC provides a good model of brain dynamics.

```{figure} ../img/rc-biology.jpg
---
width: 100%
---
Reservoirs are useful in computational neuroscience. Source: Cayco-Gajic, N. A., and Silver, R. A. (2019). Re-evaluating Circuit Mechanisms Underlying Pattern Separation. Neuron 101, 584–602. doi:10.1016/j.neuron.2019.01.044.
```

## Taming chaos (optional)


In classical RC networks, the recurrent weights are fixed and only the readout weights are trained.
The reservoir dynamics are fixed by the recurrent weights, we cannot change them.
Dynamics can be broken by external perturbations or high-amplitude noise.
The **edge of chaos** is sometimes too close.
If we could learn the recurrent weights, we could force the reservoir to have fixed and robust trajectories.

```{figure} ../img/rc-sussillo1.png
---
width: 50%
---
Classical RC networks have fixed recurrent weights. Source: {cite}`Sussillo2009`.
```

Below a classical network is trained to reproduce handwriting.
The two readout neurons produce a sequence of $(x, y)$ positions for the pen.
It works quite well when the input is not perturbed.
If some perturbation enters the reservoir, the trajectory is lost.

![](../img/rc-buonomano1.png)


We have an error signal $\mathbf{t}_t - \mathbf{z}_t$ at each time step.
Why can't we just apply backpropagation (through time) on the recurrent weights?

$$\mathcal{L}(W, W^\text{OUT}) = \mathbb{E}_{t} [(\mathbf{t}_t - \mathbf{z}_t)^2]$$

```{figure} ../img/rc-sussillo2.png
---
width: 50%
---
Learning the recurrent weights can stabilize trajectories at the edge of chaos. Source: {cite}`Sussillo2009`.
```

BPTT is too unstable: the slightest weight change impacts the whole dynamics.
In **FORCE learning** {cite}`Sussillo2009`, complex optimization methods such as **recursive least squares** (RLS) have to be used.
For the readout weights:

$$
    \Delta W^\text{OUT} = - \eta \, (\mathbf{t}_t - \mathbf{z}_t) \times P \times \mathbf{r}_t
$$

where $P$ is the inverse correlation matrix of the input :

$$
    P = (\mathbf{r}_t \times \mathbf{r}_t^T)^{-1}
$$

```{figure} ../img/RC-results.png
---
width: 70%
---
FORCE learning allows to train readout weights to reproduce compley or even chaotic signals. Source: {cite}`Sussillo2009`.
```

For the recurrent weights, we need also an error term {cite}`Laje2013`. 
It is computed by recording the dynamics during an **initialization trial** $\mathbf{r}^*_t$ and force the recurrent weights to reproduce these dynamics in the learning trials:

$$
    \Delta W = - \eta \, (\mathbf{r}^*_t - \mathbf{r}_t) \times P \times \mathbf{r}_t
$$


$P$ is the correlation matrix of recurrent activity. See <https://github.com/ReScience-Archives/Vitay-2016> for a reimplementation.

This allows to stabilize trajectories in the chaotic reservoir (**taming chaos**) and generate complex patterns even in the presence of perturbations.


```{figure} ../img/buonomano2.png
---
width: 70%
---
FORCE learning allows to train recurrent weights. Source: {cite}`Laje2013`.
```

```{figure} ../img/buonomano1.png
---
width: 100%
---
FORCE learning stabilizes trajectories. Source: {cite}`Laje2013`.
```


