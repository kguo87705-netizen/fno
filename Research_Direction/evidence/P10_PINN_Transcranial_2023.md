# P10_PINN_Transcranial_2023.pdf

## PDF页 1

Ultrasonics 132 (2023) 107026
Available online 28 April 2023
0041-624X/© 2023 Elsevier B.V. All rights reserved.
Physics-informed neural networks for transcranial ultrasound 
wave propagation 
Linfeng Wang
a
, Hao Wang
a
, Lin Liang
b
, Jian Li
a
, Zhoumo Zeng
a
, Yang Liu
a , * 
a
State Key Laboratory of Precision Measuring Technology and Instruments, Tianjin University, Tianjin 300072, China 
b
Schlumberger-Doll Research, One Hampshire St, Cambridge, MA 02139, USA  
ABSTRACT  
Transcranial ultrasound imaging has been playing an increasingly important role in the non-invasive treatment of brain disorders. However, the conventional mesh- 
based numerical wave solvers, which are an integral part of imaging algorithms, suffer from limitations such as high computational cost and discretization error in 
predicting the wavefield passing through the skull. In this paper, we explore the use of physics-informed neural networks (PINNs) for predicting the transcranial 
ultrasound wave propagation. The wave equation, two sets of time snapshots data and a boundary condition (BC) are embedded as physical constraints in the loss 
function during training. The proposed approach has been validated by solving the two-dimensional (2D) acoustic wave equation under three increasingly complex 
spatially varying velocity models. Our cases demonstrate that due to the meshless nature of PINNs, they can be flexibly applied to different wave equations and types 
of BCs. By adding physics constraints to the loss function, PINNs can predict wavefields far outside the training data, providing ideas for improving the generalization 
capability of existing deep learning methods. The proposed approach offers exciting perspectives because of the powerful framework and simple implementation. We 
conclude with a summary of the strengths, limitations and further research directions of this work.   
1. Introduction 
As an emerging diagnostic technique, transcranial ultrasound im -
aging has drawn increasing attention in the fields of stroke, cerebro -
vascular disease [1] , and even Alzheimer ’ s [2] and Parkinson ’ s disease 
[3] . It has become an attractive health care solution for being real-time, 
non-invasive, low cost and highly portable compared with other imag -
ing modalities [4] . The use of computational ultrasound models has the 
potential to predict the ultrasound field as it propagates through the 
skull in the brain, thus explaining object-specific dose and targeting 
changes [5] . Therefore, the ability to simulate the propagation of ul -
trasound through human body is critical to the approval and success of 
clinical applications. The wave equation is one of the most widely 
studied hyperbolic partial differential equations (PDEs) in scientific and 
medical applications because of its excellent illustrative capabilities [6] . 
An effective and accurate approach to transcranial ultrasound modeling 
can significantly reduce the time and money costs for patients. It also 
plays a major role in the design and customization of medical ultrasound 
transducers [7] . Therefore, it is important to accurately model the 
transcranial ultrasound propagation by solving the PDEs. 
In the literature, PDEs are commonly solved by traditional methods 
such as finite element method (FEM), finite volume method (FVM), and 
finite difference method (FDM), which have been successfully applied in 
nondestructive testing [8,9,49,50] , medical imaging [10] , metal mate -
rials [11] , geophysical [12] and other fields. However, these mesh-based 
methods suffer from the same limitations. They share the same 
computational concept and require a grid setup before solving PDEs. 
When the grid is coarsely divided, the accuracy of the solution is un -
doubtedly reduced. When the grid is very small to ensure the accuracy, it 
will greatly increase the occupation of computational resources. As the 
dimensionality of the problem increases, the well-known curse of 
dimensionality problem arises. What ’ s worse, it becomes more difficult 
for these mesh-based PDEs solvers to deal well with domains having 
irregular topography because of the poor quality of meshing [13] . These 
are considered from the perspective of spatial meshing. In addition, the 
errors caused by discretization based on temporal meshing cannot be 
ignored [14] . The PDEs are solved through time-stepping in these 
methods. In order to meet the resolution of the system, the size of the 
temporal grid needs to be controlled to improve the time sampling. 
Given the constraints of computing resources, there is no guarantee that 
this is always possible for sufficiently small spatial and temporal grid. It 
is not always easy to find a balance between solution accuracy and 
computational cost. Therefore, it is of great significance to find an 
alternative method to obtain the solution of PDEs. 
In the past few years, numerical simulation work for ultrasound wave 
propagation has never stopped developing. This has gradually led to a 
* Corresponding author. 
E-mail address: ultrasonicslab@tju.edu.cn (Y. Liu).  
Contents lists available at ScienceDirect 
Ultrasonics 
journal homepag e: www.else vier.com/loc ate/ultras 
https://doi.org/10.1016/j.ultras.2023.107026 
Received 30 October 2022; Received in revised form 9 April 2023; Accepted 24 April 2023   

## PDF页 2

Ultrasonics 132 (2023) 107026
2
deeper understanding of propagation mechanisms and advances in im -
aging techniques in various fields including medical applications. Along 
with recent advancements, neural networks (NNs) have emerged as a 
new tool for dealing with PDEs. Solving PDEs by NNs offers several 
advantages. For example, the solutions thus obtained are continuous and 
differentiable, so that we can easily get other physical parameters by 
performing integral or differential calculations. Solutions with high 
applicability over the entire domain can be obtained accurately with a 
small number of training points. In addition, a relatively simple modi -
fication of the network can cope with different boundary conditions 
(BCs) of the equations [15] . Hughes et al. [16] designed a recurrent 
neural network (RNN) structure based on the physical wave principle 
that can perform vowel recognition according to the information of 
different vowels after passing through an inhomogeneous medium. 
Moseley et al. [17] presented two types of NNs that can simulate the 
propagation of seismic waves in horizontally layered and faulted two- 
dimensional (2D) acoustic media, using WaveNet and convolutional 
autoencoder as frameworks, respectively. The convergence of the net -
works was improved by introducing a time-dependent gain term in the 
loss function. Both networks are even hundreds of times faster than 
FDM. Gantala et al. [18] proposed a Data driven-simulation-assisted- 
Physics learned AI (DPAI) model based on convolutional long short- 
term memory (ConvLSTM) and an encoder-decoder. The DPAI was 
trained using a dataset generated by FEM and successfully simulated the 
propagation of transient ultrasonic waves in a 2D domain. Ahmed et al. 
[19] also proposed a DSWE network using ConvLSTM and a convolu -
tional encoder to construct Young ’ s modulus maps based on the velocity 
data generated by a single acoustic radiation force push. However, most 
of the data-driven NNs rely on establishing nonlinear relationships be -
tween the inputs and outputs. They often require a large amount of 
labeled data for training. Because of this, these methods are relatively 
dependent on the training set, and it is a challenge to simulate accurately 
for those outside of the training distribution [17] . Stanziola et al. [20] 
developed a UNet-based fast iterative solver of the 2D heterogeneous 
Helmholtz equation with a learnable hidden state for modeling the 
transcranial wavefield. The trained solver not only performs well on the 
test set, but more importantly, still generalizes well beyond the training 
set for much larger solution domains, more complex excitation sources 
and velocity distributions. However, this method still needs to establish 
a large training set of data, which is time-consuming. Moreover, this 
method cannot obtain the time domain signal of the wavefield at arbi -
trary times. 
Physics-informed neural networks (PINNs) [21] , one of the latest 
achievements in recent years, is an approach that essentially embeds 
physical constraints into the loss function as an alternative to purely 
data-driven traditional NNs [22] . Furthermore, the automatic differen -
tiation technology in the deep learning software package is also an 
important factor for the rapid development of PINNs. This technology 
makes it very convenient to obtain the gradient of network parameters 
and further differentiate the network weight to train the network [23] . 
In PINNs, automatic differentiation technology is mainly used on the 
differentiation of the NN ’ s output to the input, and the result is taken as 
a part of the loss function. Besides, the errors in terms of the initial and 
boundary conditions (ICs/BCs) are also components of the loss function. 
These error terms together constrain the training of the network to solve 
the equation. An obvious advantage of this meshless approach is that it is 
easy to implement and can be simply extended to arbitrarily complex 
computational domains and high-dimensional PDEs, which is a chal -
lenge for conventional mesh-based approaches [24] . The rise of PINNs 
provides new ideas for the development of a large number of follow-up 
work. Song et al. [13] used PINNs to obtain the scattered pressure 
wavefields of a transversely isotropic (TI) medium with a vertical axis of 
symmetry (VTI) in a very short time and validated the method on 2D and 
three-dimensional (3D) inhomogeneous models with irregular topog -
raphy. Karimpouli et al. [25] obtained solutions of the one-dimensional 
(1D) time-dependent seismic wave equation with high accuracy from a 
small amount of data, and proved that any other parameters involved in 
the equation can also be gained by PINNs. Mao et al. [26] used PINNs to 
solve both the forward and inverse problems of the Euler equation for 
high-speed aerodynamic flows in 1D and 2D domains, and illustrated the 
superiority of PINNs in certain specific inverse problems. Rautela et al. 
[27] presented a physical knowledge assisted machine learning method 
(PKaML) and achieved damage detection and location of a carbon fiber- 
reinforced polymer plate. PKaML showed better performance than the 
direct deep learning method in terms of training time and accuracy in a 
random noise environment. More importantly, it can reduce the number 
of sensors required in actual applications. Van et al. [28] trained PINNs 
to fit the perfusion magnetic resonance (MR) data under the physical law 
of multi-compartment exchange model, and validated the method 
in silico and in vivo . Chen et al. [29] introduced the PINN-SR method 
(PINNs with sparse regression), which can predict the key parameters in 
the governing equations of nonlinear systems from sparse and noisy 
data. Raissi et al. [30] developed a hidden fluid mechanics (HFM) based 
on the framework of PINNs, which addressed the problem of flow 
visualization by embedding the Navier-Stokes equation into the net -
works. Lu et al. [31] solved different types of equations using DeepXDE 
and improved the training efficiency of PINNs by applying a new 
residual-based adaptive refinement (RAR) method. PINNs have shown 
excellent performance in many forward and inverse problems in the 
fields of science and engineering. However, few studies have been 
conducted regarding their use with transcranial ultrasound wave 
propagation. 
In this paper, we first propose an approach for transcranial ultra -
sound wave propagation using PINNs. We extend the original PINNs 
method to predict the wavefield of transcranial ultrasound at any 
moment and any position, requiring two early-time snapshots as training 
data. In this study, the snapshots in terms of displacement are obtained 
by the spectral-element method (SEM). It is worth noting that conven -
tional PDEs solvers cannot simulate transcranial ultrasound propagation 
while using only two early-time wavefield snapshots, which highlights 
the superiority and flexibility of PINNs in solving the wave equation 
with a small amount of data. Furthermore, the numerical solutions ob -
tained by SEM are considered as the reference solutions to evaluate the 
effectiveness and accuracy of the prediction results of PINNs. The per -
formance of the proposed method has been verified on a homogeneous 
model, an inhomogeneous model and a 2D brain model with and 
without a clot, respectively. For the first two cases, it has been proved 
that the proposed method has impressive generalization capability and 
can match the wavefield solutions well given only two early-time 
snapshots. For the brain model, PINNs can also predict most informa -
tion including reflected wave when propagates to the skull. Although 
there is a loss of information of the predictions inside the brain, the time 
domain signals in the × direction and y direction at the receiving sensors 
indicate that these losses are acceptable. Besides, it is not difficult to 
apply the algorithm to any complex computational boundary because of 
the meshless nature of PINNs. The following contents of this paper are as 
follows: a brief overview of the methodology for transcranial ultrasound 
wave propagation and PINNs is described in Section II. Section III carries 
out numerical simulations under different conditions and the results are 
illustrated. The advantages and limitations of the proposed method are 
discussed in Section IV and final conclusions are summarized in Section 
V. 
2. Methodology 
2.1. 2D acoustic transcranial ultrasound wave equation 
In general, the wave propagation of transcranial ultrasound can be 
very complex due to many issues such as inhomogeneous velocity and 
density distributions, shear wave effects in the skull, nonlinear effects, 
and different acoustic absorption coefficients involved [32,33] . How -
ever, when the ultrasound is propagating through the skull close to the 
L. Wang et al.                                                                                                                                                                                                                                   

## PDF页 3

Ultrasonics 132 (2023) 107026
3
normal incidence, the shear wave effects can be ignored [34,35] . In 
addition, the nonlinear effect is mainly used in ablation treatment, and is 
only effective in a small range near the lesion [36] . Therefore, this 
process can be described by the following linear acoustic wave equation 
in an inhomogeneous absorbing medium after simplification [35,37] : 
(
1 + τ
0
( r )
∂
∂ t
)[
ρ
0
( r )∇ ⋅
(
1
ρ
0
( r )
)
∇ p ( r , t )
]
 
1
c
0
( r )
2
∂
2
p ( r , t )
∂ t
2
= S
0
( r , t ) , (1) 
where τ
0 
represents the spatial distribution of the absorption coef -
ficient, r is a general space coordinate, ρ
0 
stands for the spatial distri -
bution of the density, p is the pressure in the acoustic medium, c
0 
and S
0 
represent the distribution of the wave velocity and the acoustic source. 
The above equation accurately describes the law of wave propagation 
when the velocity, density and absorption coefficient are inhomoge -
neously distributed. If the acoustic absorption is treated as a second- 
order effect [38] , Eq. (1) describing the wave propagation of trans -
cranial ultrasound can be further simplified as [20] : 
ρ
0
( r )∇ ⋅
(
1
ρ
0
( r )
)
∇ p ( r , t )  
1
c
0
( r )
2
∂
2
p ( r , t )
∂ t
2
= S
0
( r , t ) . (2) 
Eq. (2) can be expressed as Eq. (3) with the scalar wave potential ϕ : 
∇
2
ϕ  
1
c
2
0
∂
2
ϕ
∂ t
2
= S
0
, (3) 
where ∇
2
=
∂
2
∂ x
2
+
∂
2
∂ y
2 
is the Laplace operator in the 2D Cartesian co -
ordinate system. 
Although the wave equation is linear, it is well known that due to the 
oscillatory, broadband, and dispersion of the solution, it is never easy to 
solve it in complex medium [22] . The physical laws involved in this 
wavefield can also be very complex, requiring consideration of phe -
nomena including reflection, refraction, transmission and scattering at 
the junction of the different velocity distributions or the boundary of the 
domain, the simultaneous interaction of multiple types of waves, and the 
corresponding change in amplitude as the wave propagates [39] . The 
fact that the skull is solid – liquid coupled, multi-layered and porous, 
coupled with the nature of the velocity of the skull being much faster 
than the internal tissues, increases the difficulty of modeling the prop -
agation of transcranial ultrasound. Therefore, the problem may be even 
more difficult to solve than some nonlinear problems that have been 
addressed with PINNs, such as fluid mechanics. 
2.2. The principle of PINNs 
It has been proved that the Feed-forward neural network (FNN) is 
sufficient to solve PDEs [15] . This is one of the reasons why the archi -
tecture of PINN is almost completely based on a fully connected FNN in 
the current literature. In this paper, we define a fully connected FNN 
with three channels of inputs: the spatial coordinates x , y and time t . The 
output of the network is the predicted value of the scalar wave potential 
ϕ and the network has L hidden layers. The displacement variable in the 
× and y directions ( u
x 
and u
y
) during wave propagation can be derived 
by automatic differentiation of ϕ : 
u
x
=
∂ ϕ
∂ x
(4)  
u
y
=
∂ ϕ
∂ y 
In FNN, the value of the current layer is calculated from the previous 
one as: 
q
i
= σ
i
 
ω
i
x
i
+ b
i
)
, i = 1 , 2 , ⋯ , L (5) 
where σ represents the activation function, such as sine function, 
hyperbolic tangent activation function and sigmoid function, and q 
stands for the values of i
th 
layer. The training of FNN is essentially an 
optimization process. In each iteration, the weights ω and biases b of the 
network are updated by minimizing the loss function. Mean Square 
Error (MSE) is one of the most commonly used loss functions, which is 
defined as follows: 
MSE =
1
N
u
∑
N u
i = 1
( u   u
*
)
2
, (6) 
where u and u
* 
are the reference solution and the predicted value 
output by the network and N
u 
stands for the number of points selected to 
calculate the MSE . Then we sample a certain number of training points 
from two early-time snapshots and boundary data to train FNN, making 
it satisfy Eq. (3) at the same time. In fact, the outputs of the network must 
meet the PDE, the two snapshots, and BC to achieve the best results. 
Therefore, the training process can be transformed into minimizing the 
weighted sum of the MSEs : 
MSE ( Θ ) = λ
PDE
MSE
PDE
+ λ
S 1
MSE
S 1
+ λ
S 2
MSE
S 2
+ λ
BC
MSE
BC
, (7) 
where Θ = ω
⋃
b stands for the union of all the weights and biases in 
the network. MSE
PDE
, MSE
S 1
, MSE
S 2
, and MSE
BC 
represent the PDE error 
function, the first snapshot error function, the second snapshot error 
function and BC error function using in Eq. (6) respectively. In this paper, 
the IC is regarded as a kind of BC. λ
PDE
, λ
S 1
, λ
S 2
, and λ
BC 
are the weights of 
the error functions that are adjusted according to different situations to 
converge to the correct solution. Improper weighting of loss functions 
will slow down the convergence speed of the network training, and even 
affect the prediction accuracy seriously. Compared with the conven -
tional PDEs solvers, the proposed method predicts the solution by 
solving an optimization problem which is carried out by updating the 
weight and biases from each iteration to minimize Eq. (7) , i.e., 
Θ
*
= argmin { MSE ( Θ
*
) } . (8) 
We use Adam as the optimization function [40] with a suitably 
chosen batch size (BSGD) for all the cases in this paper: 
( ω
next
, b
next
) = ( ω
current
, b
current
)   ξ
∂ MSE
∂ ( ω
current
, b
current
)
, (9) 
where ξ is the learning rate during training. Training stops when the 
network iterations reach saturation or when the training epoch reaches a 
set value. 
Unlike traditional numerical methods, since the solutions of PINNs 
are obtained by solving a non-convex optimization problem, there is no 
guarantee of unique solutions for PINNs. In practice, we need to tune all 
the hyperparameters to achieve a satisfactory level of accuracy. The 
required network size depends largely on the smoothness of the solution 
of PDEs. The setting of hyperparameters in the network cannot be 
neglected, which will affect the convergence of PINNs to different so -
lutions [31,41] . Therefore, we need to train PINNs for a few times with 
different hyperparameters and select the network with the least training 
loss as the final solution. More importantly, a normalization step is 
required before training. It consists of mapping the input and output in 
the interval [   1 , 1 ] ∈ R and scaling Eq. (3) to constrain the wave speed in 
the interval [ 0 , 1 ] ∈ R . 
2.3. PINNs design and workflow 
Fig. 1 shows the workflow of PINNs for transcranial ultrasound wave 
propagation. The input of the network is the spatial coordinates x , y and 
time t , and the output is the prediction of the wavefield for this coor -
dinate at this moment. According to the complexity of the medium, we 
adjust the number of hidden layers and neurons of the network to ach -
ieve the best convergence result. We use the identity function as the 
activation function between the last hidden layer and the output, and 
σ = tanh (•) function is chosen for the rest of the hidden layers. In this 
paper the early-time snapshots in terms of displacement used for 
L. Wang et al.                                                                                                                                                                                                                                   

## PDF页 4

Ultrasonics 132 (2023) 107026
4
training are obtained by the SEM. After setting the parameters such as 
velocity model, density model, location and frequency of excitation 
source as ICs, we carry out forward simulation with a few time steps to 
obtain the reference solutions of early wavefields. Importantly, only two 
early-time snapshots are used to train the network. After training, we use 
the wavefield at later time steps from forward simulation to test the 
generalization ability of PINNs. For each update step, a set of points are 
sampled from the training data to calculate the MSEs in Eq. (7) . A batch 
size of 2000 is set for each update step with a learning rate ξ of 3 × 10
-4 
in all cases. 
3. Results 
We used three increasingly complex velocity models in Fig. 2 to 
evaluate the predictive ability of the proposed method for the propa -
gation of transcranial ultrasound. It is necessary to reset the parameters 
of FNN and the weights in Eq. (7) to retrain PINNs for each velocity 
model. The models include: (1) a homogeneous model with a fixed 
sound velocity of 1500 m / s ; (2) an inhomogeneous model including 
1500 m / s and 2000 m / s and (3) a 2D brain model, the outer part of 
which is considered to be surrounded by water with a velocity of 1500 
m / s [42] . The multimodal imaging-based detailed anatomical (MIDA) 
model is a segmented cranial phantom which has been created from a 
combination of scans taken with magnetic resonance imaging, magnetic 
resonance angiography, and diffusion tensor imaging [43] . We used the 
the human head data in MIDA model as the original data sample. 465 
slices were obtained with aninterval of 0.5 mm, and one of them was 
selected to build the 2D brain model insilico simulations. The data in 
MIDA model is separated according to tissue types, and each has its 
unique physical properties. The data also combines the sound velocity 
and density of different tissue types from [33 – 37,44] . However, the 
acoustic characteristics of small tissues may be difficult to measure in 
practice (especially invivo ), resulting in little or no recording of the 
acoustic characteristic parameters of these tissues. In these cases, we use 
parameters from other tissues similar in structure and composition to the 
tissue for estimation, provided that such estimation will not affect the 
wave propagation. Fig. 2 (d) shows the brain slice selectedfrom MIDA, 
which can be used for generating reference solutions and PINNs training 
after being transformedinto the 2D brain velocity model ( Fig. 2 (c)). 
3.1. Homogeneous model 
In this case, we explore the ability of PINNs under a homogeneous 
model as shown in Fig. 2 (a). We choose a Gaussian point source with a 
dominant frequency of 20 kHz at (0.5,0.8) as the excitation source. 
There are 7 hidden layers in FNN and each contains 40 neurons. In the 
process of propagation with a constant velocity, the amplitude of the 
wave will gradually decrease due to the spherical dispersion. In this 
case, all the surfaces are absorbing boundaries except for free-surface 
condition for the top of the domain. For MSE
PDE 
term, we randomly 
sample 10,000 points for training, while for two early-time snapshots, 
we sample 6400 points respectively. Another 5000 points are needed to 
train MSE
BC 
terms, due to the free-surface condition for the top surface. 
The weight values for different MSE terms in Eq. (7) are λ
PDE
= 0 . 1, λ
S 1
=
λ
S 2
= 1, and λ
BC
= 0 . 01 . When the loss items are almost no longer 
reduced or have reached the arranged value of epoch, we consider the 
network training to be over. The time step is set to 5 × 10
  8
s , and we 
simulate 0.5 ms for wave propagation. Fig. 3 (a1) to (a4) show the 
reference solutions obtained by the SEM at 0.10 ms , 0.12 ms , 0.33 ms and 
0.45 ms , respectively. Among them, only the first two early-time snap -
shots are used for training. Fig. 3 (b1) to (b4) represent the predictions of 
PINN at corresponding times. Fig. 3 (c1) to (c4) represent the errors 
between them. The prediction results agree with the reference solutions 
accurately, capturing both its kinematics and amplitude attenuation. It 
is important to highlight that no other labeled data were used 
throughout the training process except for two early-time snapshots that 
contained information about the excitation source. After setting the BC 
loss term, PINN successfully applies the BCs to the corresponding sur -
faces, capturing the reflected waves from the top surface as shown in 
Fig. 3 (b3) and (b4). Although the training data does not contain infor -
mation about the reflections, PINN can still accurately predict the po -
sition and amplitude of the reflected waves with the maximum error 
being less than 3%. In addition, since the solution of PINN is smooth and 
infinitely differentiable, it can automatically satisfy the absorbing BC, as 
shown in Fig. 3 (b4). We can effectively choose different BCs for PINN by 
setting the BC loss term, which is exactly what we want to demonstrate 
through this case. After offline training, we can obtain the time domain 
signal of the point by inputting the space coordinates x and y online, or 
the wavefield of the whole solution domain at that time by inputting the 
time t online. It is worth noting that this process only takes about 3 s on a 
single CPU workstation. PINN can accurately predict the wavefield at 
times much later than the training wavefield, showing satisfying 
generalization capability outside of the training data. Fig. 4 shows the 
convergence of different loss terms. The errors still show a decreasing 
trend, indicating that the difference between the reference solution and 
prediction is still possible to be smaller. Although we trained 380,000 
epochs for this case, PINN has converged to a relatively accurate result 
Fig. 1. Schematics of PINNs ’ workflow for transcranial ultrasound wave 
propagation. We think the training stops when MSE becomes smaller than the 
threshold ( ∊ ), or the training reaches the set value (max_epoch). 
L. Wang et al.                                                                                                                                                                                                                                   

## PDF页 5

Ultrasonics 132 (2023) 107026
5
in less than 150,000 epochs, and the training process later is used to 
improve the subtle errors. This is explained and illustrated in the 
following case of 2D brain model. 
3.2. Inhomogeneous model 
In this case, we explore the ability of PINNs under an inhomogeneous 
model with two kinds of velocity as shown in Fig. 2 (b) which induces 
complex wavefield dynamics. In the process of propagation, the wave -
field may be compressed or expanded as the velocity changes, and the 
reflected and transmitted waves will be formed at the interfaces of 
different velocities. We choose a point source with a dominant frequency 
of 30 kHz and a Gaussian source time function located at the center of 
the domain to excite the initial wavefield. Due to the increase of exci -
tation frequency and the complexity of velocity model, we adjust the 
number of hidden layers to 8 and the number of neurons in each to 50. 
The training points of MSE
PDE 
is increased to 20,000 and we remain 
6400 points for each of the early-time snapshots data. In this case, we set 
absorbing conditions for all the boundaries. The weights of the misfit 
function are λ
PDE
= 0 . 1, λ
S 1
= λ
S 2
= 1 and λ
BC
= 0. We simulate 0.5 ms 
wave propagation with the time step set to be 5 × 10
  8 
s . Fig. 5 shows 
the reference solutions run by SEM, the wavefields predicted by PINN 
after training and the differences between them. The wavefield solutions 
at 0.06 ms , 0.09 ms , 0.23 ms and 0.30 ms are compared respectively. 
Again, only two early-time snapshots were used for PINN training. After 
offline training we can obtain the time domain signal of the input co -
ordinate ( x , y ) , or the wavefield at the input time t . For the convenience 
of observation, different colormaps are used for the wavefields of the 
first two and the last two moments. From (a3) and (a4) of Fig. 5 , we can 
see that although the velocity model is relatively simple, its corre -
sponding wavefield is indeed complex, with multiple transmissions and 
reflections occurring at the interface where the velocity changes sharply. 
Still, the proposed method can reasonably capture most of the trans -
mitted and reflected waves which are not available in the training data. 
It is also able to describe the phenomena of compression, expansion and 
amplitude attenuation of the wavefields. The maximum error is about 
8% in this model. It is worth noting that only two early-time snapshots 
are used for training, and at these two moments the waves have not yet 
propagated to different velocity interfaces. That is, the constraints of the 
PDE play an indispensable role in the training process. However, by 
observing the differences in more detail, we find that the loss along the 
interface is a bit higher in the velocity model. One explanation may be 
that the discontinuity in the velocity model leads to the discontinuity of 
the second order gradients of the wavefields in the network, which is 
difficult to learn [22] . Fig. 6 shows the convergence of different loss 
terms. 
Fig. 2. Velocity models. (a) a homogeneous model; (b) an inhomogeneous model; (c) a 2D brain model transformed from (d), a slice selected from MIDA. Specific 
networks need to be built for each velocity model. The red circles indicate the positions of the excitation sources in the cases. (For interpretation of the references to 
color in this figure legend, the reader is referred to the web version of this article.) 
L. Wang et al.                                                                                                                                                                                                                                   

## PDF页 6

Ultrasonics 132 (2023) 107026
6
3.3. 2D brain model 
Our final case uses the 2D brain model shown in Fig. 2 (c), which 
produces a much more complex wavefield. A Gaussian point source is 
located at (0.25, 0.1) with a central frequency of 100 kHz. An FNN with 
10 hidden layers and 60 neurons in each layer is selected considering the 
complexity of the model and the high frequency. The training data sets 
for PDE and each of the early-time snapshots are composed of 60,000 
and 10,000 data points, respectively. We impose an absorbing condition 
for all the surfaces of the domain. At the same time, the weights of the 
loss functions in Eq. (7) are adjusted to λ
PDE
= 0 . 01, λ
S 1
= λ
S 2
= 1 and 
λ
BC
= 0. The time step is set to 2 × 10
  8
s , and we run 0.2 ms for wave 
propagation using the SEM. Fig. 7 shows the reference solutions, the 
predictions and the differences at 0.036 ms , 0.044 ms , 0.092 ms and 
0.144 ms , respectively. The wavefields of Fig. 7 (a1) and (a2) are used for 
training. Again, (a3) and (a4) are obtained by online testing. For the 
convenience of observation, the 2D brain velocity model is added to 
each result as a background, which is independent of the colormap 
corresponding to each result. The predictions for the first two moments 
match the reference solutions quite well, with a maximum error less 
than 5%. For the latter two moments, PINN can also predict the infor -
mation of the reflection as it propagates to the skull in the brain model. 
However, when the wavefront propagates through the brain model, 
there is a loss of information of the predictions about the wavefield in -
side the brain. This may be due to excessive velocity differences between 
the skull and internal tissues and the fact that the amplitude of the in -
formation is too small to be accurately modeled by PINN. Fig. 8 shows 
the training loss for the 2D brain model against training step. 
To understand the effect of training iterations, we recorded the 
predictions of PINN for the wavefield of 0.144 ms in this case at different 
epochs. Fig. 9 (a1) to (a4) show the prediction results of PINN at 12,000, 
20,000, 40,000 and 60,000 epochs, respectively, and (b1) to (b4) are the 
differences between them and the reference solution as shown in Fig. 7 
(a4). Despite the complexity of the model, PINN begins to predict tiny 
contours of the wavefield at 12,000 epochs and has roughly converged 
by 20,000 epochs. At 60,000 epochs, PINN can predict the reflected 
information of the wavefield. The subsequent training is used to adjust 
the subtle wavefield information and stops at 100,000 epochs. 
To further test the feasibility of PINN in predicting transcranial ul -
trasound wave propagation, we modified the 2D brain model to contain 
a clot, as shown in Fig. 10 (a). We use physical characteristics for blood- 
infused soft tissue to establish this model [45] . The clot is defined as a 
circular area with a radius of 15 mm and a fixed sound velocity of 1600 
Fig. 3. Comparison of the PINN ’ s wavefield predictions to the reference solutions using the homogeneous model. All the surfaces are absorbing boundaries except for 
free-surface condition for the top in this domain. (a1) to (a4) and (b1) to (b4) show the reference solutions and predictions through time. (c1) to (c4) show the 
differences between them. We unify the color bar ranges in order to reflect the amplitude attenuation in the process of wave propagation. 
Fig. 4. The convergence of training loss for the homogeneous model.  
L. Wang et al.                                                                                                                                                                                                                                   

## PDF页 7

Ultrasonics 132 (2023) 107026
7
m / s . Compared with the previous case, all the conditions are completely 
consistent except the velocity model (with or without the clot). (b) 
shows the reference solution obtained by the SEM at 0.144 ms with this 
model and (c) is the prediction of PINN at this moment. The difference 
between (b) and (c) is depicted in (d). Overall, the PINN ’ s prediction 
matches the reference solution reasonably well. Similar to the case 
without the clot, the error is mainly reflected in the wavefront as it 
passes through the brain. Fig. 11 shows the differences of the wavefields 
between 2D brain models with and without the clot at different times 
obtained by the SEM. They can be regarded as the wavefields changing 
over time with the clot used as the source. 
The match between the reference solutions and the PINN ’ s pre -
dictions of the motion on × and y axial for the 2D brain model with the 
clot is shown in Fig. 12 . Locations of the receiving sensors are shown as 
the green triangles in Fig. 10 (a). They are evenly distributed between 
(0.40,0.10) and (0.40,0.30). We can see there are two peaks in the ×
axial motion, of which the front one represents the signal propagating 
forward and the other one represents the reflected signal. For some of 
the reflected signals, there are some discrepancies between the reference 
solutions and predictions. However, for the fluctuating trend of the time 
domain signal, the solution of PINN closely matches the reference, 
which proves its ability of calculating the wavefield. Fig. 13 shows the 
comparison of the spectrum on × and y axial at (0.40,0.30). The spec -
trum of the prediction is basically consistent with the reference, with the 
center frequency near 100 kHz. 
4. Discussion 
4.1. Advantages of this method 
We have demonstrated that PINNs can solve the problem of wave 
propagation for transcranial ultrasound by using several increasingly 
complex spatial velocity models. The complexity of the network (such as 
the number of hidden layers and neurons) increases with the complexity 
of the problem (such as frequency and velocity models) [46] . For the 
same velocity model, modeling for higher frequencies requires more 
PDE training points to capture finer features, such as multiple reflected 
waves. PINNs are mesh-free and consequently can be extended to arbi -
trarily complex computational boundaries with relative ease. Impres -
sively, they can predict subtle physical phenomena such as transmitted 
and reflected waves at different velocity interfaces, wavefield 
compression and expansion, and amplitude attenuation of wavefields at 
much later times than the training data, which is a promising approach 
among existing deep learning approaches. Compared with traditional 
methods, this deep learning-based method improves online prediction 
by transferring excessive computational load to offline by training a NN 
model. After the network training is completed, we use PINN and SEM to 
generate ten wavefields successively on the same device (a single CPU 
Fig. 5. Comparison of the PINN ’ s wavefield predictions to the reference solutions using the inhomogeneous model. All the surfaces are absorbing boundaries.  
Fig. 6. The convergence of training loss for the inhomogeneous model.  
L. Wang et al.                                                                                                                                                                                                                                   

## PDF页 8

Ultrasonics 132 (2023) 107026
8
workstation) to compare the computation time. It should be noted that 
we set the same time step in advance. Table 1 shows the comparison 
results for the three models. For all the three models, the computation 
time of PINN is smaller than that of SEM. In addition, it is also notable 
that when the model becomes complex, the computation time of PINN is 
not greatly affected, while the required time of SEM changes signifi -
cantly. Meanwhile, PINN can directly obtain the wavefield at the later 
time, while SEM must iterate from the initial time to the later time. 
These clearly demonstrate the superiority of PINN in computation. 
In addition, since the method is meshless, it avoids the dispersion 
problem that traditional mesh-based methods typically encounter when 
the mesh is coarse. PINNs could also be used for source inversion, as it is 
readily accessible to obtain the gradients of the input terms and it is fully 
flexible in the selection of measurement positions for the observed 
wavefield. Another benefit of this approach is that we can quantify how 
well the network satisfies the corresponding physical constraints 
through each loss term, which helps to evaluate the accuracy and reli -
ability of the network and improve its performance [22] . 
4.2. Limitations and solutions 
In this work we find that modeling at the interface with high velocity 
contrast is the main cause of errors because the second-order derivatives 
of the wavefields at these locations change discretely, which is a chal -
lenge for PINNs to learn. In fact, this problem is also tricky for traditional 
PDE solvers, which usually take an oversampling approach to guarantee 
the accuracy of the solutions [39] . For PINN, this situation can be 
improved to a certain extent by adjusting the training parameters such 
as network depth and number of iterations. It is also a good solution to 
choose a more general wave equation. Another related issue is that 
PINNs have difficulty in simulating reflected waves of relatively low 
amplitudes in 2D brain model. Future work could explore these issues, 
for example, by using specific networks for different velocity regions to 
reduce the impact of network discontinuities, by trying different acti -
vation functions [47] , by adding physical constraint terms to make the 
loss function more relevant to the practical problems, or by adopting 
smarter sampling schemes [48] . Admittedly, we use data from SEM as 
the training set, so the accuracy of PINN ’ s results will not be better than 
that of SEM. However, it is undeniable that PINN improves the 
computational efficiency by transferring computational load to offline 
training, and has the advantages of no dispersion, easy dealing with 
different BCs, and not requiring a large amount of label data, which are 
the aspects we would like to emphasize. The proposed method may 
bring new enlightenment to the field of transcranial ultrasound. 
In this paper, a signal with a dominant frequency of 100 kHz is 
chosen for the simulation of the transcranial wavefield. This frequency is 
relatively low. In fact, if we choose a higher frequency (e.g., 0.5 MHz), 
Fig. 7. Comparison of the PINN ’ s wavefield predictions to the reference solutions using the 2D brain model. All the surfaces are absorbing boundaries.  
Fig. 8. The convergence of training loss for the 2D brain model.  
L. Wang et al.                                                                                                                                                                                                                                   

## PDF页 9

Ultrasonics 132 (2023) 107026
9
SEM needs to divide the grid several times more to obtain a wavefield 
without dispersion. When using the snapshot data under such condition 
for PINN training, more sampling points are also required to ensure the 
accuracy of the predicting results, which is not allowed by the current 
computational power. It seems that PINNs require a higher computa -
tional cost to train than numerical solutions in some applications. When 
the velocity model or the position of the excitation source changes, PINN 
needs to be retrained. However, on the one hand, traditional methods 
also face this problem; on the other hand, PINN can improve the 
generalization capability by taking the coordinates of the excitation 
source as the input of the network, and can improve the computational 
efficiency by applying the trained network to the model with little dif -
ference through transfer learning, which has been investigated. In 
addition, we have been already working on the prediction of the 
transcranial wavefield in frequency domain by embedding the Helm -
holtz equation into the loss function. The prediction in frequency 
domain will also save computational resources compared to time 
domain. This method has the potential to pave a new way for 
Fig. 9. Comparison of the PINN ’ s wavefield predictions to the reference solution at different epochs for the 2D brain model.  
Fig. 10. Comparison of the PINN ’ s wavefield prediction to the reference solution using the 2D brain model with a clot. (a) 2D brain model with a clot (the green 
triangles represent the receiving sensors); (b) the reference solution obtained by the SEM; (c) the prediction of PINN; (d) the difference between (b) and (c). (For 
interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.) 
L. Wang et al.                                                                                                                                                                                                                                   

## PDF页 10

Ultrasonics 132 (2023) 107026
10
applications in other fields. This includes applications in more complex 
physics, like viscoelastic anisotropic medium and fluid-saturated porous 
medium, where the traditional wave equation solvers become imprac -
tical, especially in the case of large model sizes. Deep learning has the 
ability to learn nonlinear relationships in a given problem by identifying 
patterns and making accurate judgments and predictions about them, 
making it one of the best methods for dealing with large data and large 
problems. For complex physical problems, the loss function may be 
changed with the addition of more constraint terms and the adjustment 
of the weights, but the network shares the same training mechanism and 
does not need to solve the inverse of large matrices. 
4.3. Future research directions 
Finally, our approach can be easily extended to 3D and more com -
plex wave equations to model the transcranial ultrasound propagation 
more accurately. 3D extension is trivial in the network design, requiring 
only one additional dimension of the input tensor. Furthermore, 
Fig. 11. The differences of the wavefields between 2D brain model with and without a clot at different times obtained by the SEM.  
Fig. 12. The comparison between the reference solutions (blue line) and the PINN ’ s predictions (red dashed line) of the motion on × and y axial for the 2D brain 
model with a clot. Locations of the receiving sensors are shown as green triangles in Fig. 10 (a). (For interpretation of the references to color in this figure legend, the 
reader is referred to the web version of this article.) 
L. Wang et al.                                                                                                                                                                                                                                   

## PDF页 11

Ultrasonics 132 (2023) 107026
11
generating 3D training data may not be too challenging in practice, as 
the proposed approach only requires a few early-time snapshots of 
reference solutions to train the network. For more complex situations, 
such as the elastic wave equation in more complex medium, we need to 
modify the loss function term in Eq. (7) . The performance of the pro -
posed method for transcranial ultrasound wave propagation indicates 
that PINNs may also do well for other complex physical systems. Our 
future work will investigate these directions. 
5. Conclusion 
In this paper, we solve the problem of transcranial ultrasound wave 
propagation by applying PINNs to 2D acoustic wave equation. We use a 
fully connected FNN, which takes spatial coordinates as inputs and the 
wavefield solution as output to the network. The performance of PINNs 
is verified by a homogeneous model, an inhomogeneous model and a 2D 
brain model with and without a clot. The results show that the proposed 
method has an impressive generalization capability and can predict the 
wavefield solution much later in time given only two early-time snap -
shots for training. Future work includes its application to 3D brain im -
aging through 3D physical equations and the consideration of the 
viscoelastic anisotropic medium or fluid-saturated porous medium, 
which is closer to the real situation. More work needs to be done to 
extend it for more general model configurations (e.g., arbitrary velocity 
models, or velocity models with limited variations) before it can be used 
for imaging. 
Declaration of Competing Interest 
The authors declare that they have no known competing financial 
interests or personal relationships that could have appeared to influence 
the work reported in this paper. 
Data availability 
No data was used for the research described in the article. 
Acknowledgements 
This work was supported in part by the National Science Foundation 
of China under contract number 61773283, in part by National Key R & D 
Program of China under contract number 2018YFC0808600. 
REFERENCES 
[1] D. Antipova, L. Eadie, S. Makin, H. Shannon, P. Wilson, A. Macaden, The use of 
transcranial ultrasound and clinical assessment to diagnose ischaemic stroke due to 
large vessel occlusion in remote and rural areas, PLoS One 15 (10) (2020) 
e0239653 . 
[2] A.E. Roher, Z. Garami, S.L. Tyas, C.L. Maarouf, T.A. Kokjohn, M. Belohlavek, L. 
J. Vedders, D. Connor, M.N. Sabbagh, T.G. Beach, M.R. Emmerling, Transcranial 
Doppler ultrasound blood flow velocity and pulsatility index as systemic indicators 
for Alzheimer ’ s disease, Alzheimers Dement. 7 (4) (2011) 445 – 455 . 
[3] A. Gaenslen, B. Unmuth, J. Godau, I. Liepelt, A. Di Santo, K.J. Schweitzer, 
T. Gasser, H.-J. Machulla, M. Reimold, K. Marek, D. Berg, The specificity and 
sensitivity of transcranial ultrasound in the differential diagnosis of Parkinson ’ s 
disease: a prospective blinded study, Lancet Neurol. 7 (5) (2008) 417 – 424 . 
[4] M. Mozaffarzadeh, E. Verschuur, M.D. Verweij, V. Daeichin, N. De Jong, 
G. Renaud, Refraction-Corrected Transcranial Ultrasound Imaging Through the 
Human Temporal Window Using a Single Probe, IEEE Trans. Ultrason. Ferroelectr. 
Freq. Control. 69 (4) (2022) 1191 – 1203 . 
[5] W. Lee, H.-C. Kim, Y. Jung, Y.A. Chung, I.-U. Song, J.-H. Lee, S.-S. Yoo, 
Transcranial focused ultrasound stimulation of human primary visual cortex, Sci 
Rep. 6 (1) (2016) 1 – 12 . 
[6] M. Almekkawy, J. Chen, M.D. Ellis, D. Haemmerich, D.R. Holmes, C.A. Linte, 
D. Panescu, J. Pearce, P. Prakash, V. Zderic, Therapeutic systems and technologies: 
State-of-the-art applications, opportunities, and challenges, IEEE Rev. Biomed. 
Eng. 13 (2019) 325 – 339 . 
[7] J. Gu, Y. Jing, Modeling of wave propagation for medical ultrasound: a review, 
IEEE Trans. Ultrason. Ferroelectr. Freq. Control. 62 (11) (2015) 1979 – 1992 . 
[8] H. Wang, J. Li, L. Wang, L. Liang, Z. Zeng, Y. Liu, On acoustic fields of complex 
scatters based on physics-informed neural networks, Ultrasonics 106872 (2022) . 
[9] J. Tong, M. Lin, X. Wang, J. Li, J. Ren, L. Liang, Y. Liu, Deep learning inversion 
with supervision: A rapid and cascaded imaging technique, Ultrasonics 122 (2022), 
106686 . 
[10] B. Vafaeian, M. El-Rich, T. El-Bialy, S. Adeeb, The finite element method for micro- 
scale modeling of ultrasound propagation in cancellous bone, Ultrasonics 54 (6) 
(2014) 1663 – 1676 . 
[11] K. Nakahata, H. Sugahara, M. Barth, B. K ¨ohler, F. Schubert, Three dimensional 
image-based simulation of ultrasonic wave propagation in polycrystalline metal 
using phase-field modeling, Ultrasonics 67 (2016) 18 – 29 . 
[12] Z. Guo, G. Egbert, H. Dong, W. Wei, Modular finite volume approach for 3D 
magnetotelluric modeling of the earth medium with general anisotropy, Phys. 
Earth Planet. Inter. 309 (2020), 106585 . 
[13] C. Song, T. Alkhalifah, U.B. Waheed, Solving the frequency-domain acoustic VTI 
wave equation using physics-informed neural networks, Geophys. J. Int. 225 (2) 
(2021) 846 – 859 . 
[14] S. Alkhadhr, X. Liu, M. Almekkawy, Modeling of the Forward Wave Propagation 
Using Physics-Informed Neural Networks, in, IEEE International Ultrasonics 
Symposium (IUS). 2021 (2021) 1 – 4 . 
[15] S. Panghal, M. Kumar, Optimization free neural network approach for solving 
ordinary and partial differential equations, Eng. Comput. 37 (4) (2020) 
2989 – 3002 . 
[16] T.W. Hughes, I.A. Williamson, M. Minkov, S. Fan, Wave physics as an analog 
recurrent neural network, Sci. Adv. 5 (12) (2019) eaay6946 . 
Fig. 13. The comparison between the reference solutions (blue line) and the PINN ’ s predictions (red dashed line) of the spectrum on × and y axial at (0.40,0.30). 
(For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.) 
Table 1 
The computation time of PINN and SEM to obtain ten wavefields (Unit: seconds).  
Model SEM PINN 
Homogeneous model  12.91  10.25 
Inhomogeneous model  29.20  14.44 
2D brain model  41.40  16.90  
L. Wang et al.                                                                                                                                                                                                                                   

## PDF页 12

Ultrasonics 132 (2023) 107026
12
[17] B. Moseley, T. Nissen-Meyer, A. Markham, Deep learning for fast simulation of 
seismic waves in complex media, Solid Earth 11 (4) (2020) 1527 – 1549 . 
[18] T. Gantala, K. Balasubramaniam, DPAI: A Data-driven simulation-assisted-Physics 
learned AI model for transient ultrasonic wave propagation, Ultrasonics 121 
(2022), 106671 . 
[19] S. Ahmed, U. Kamal, M.K. Hasan, DSWE-Net: A deep learning approach for shear 
wave elastography and lesion segmentation using single push acoustic radiation 
force, Ultrasonics 110 (2021), 106283 . 
[20] A. Stanziola, S.R. Arridge, B.T. Cox, B.E. Treeby, A Helmholtz equation solver using 
unsupervised learning: Application to transcranial ultrasound, J. Comput. Phys. 
441 (2021), 110430 . 
[21] M. Raissi, P. Perdikaris, G.E. Karniadakis, Physics-informed neural networks: A 
deep learning framework for solving forward and inverse problems involving, 
J. Comput. Phys. 378 (2019) 686 – 707 . 
[22] B. Moseley, A. Markham, T. Nissen-Meyer, Solving the wave equation with physics- 
informed deep learning, arXiv preprint arXiv:2006.11894, 2020. 
[23] A.G. Baydin, B.A. Pearlmutter, A.A. Radul, J.M. Siskind, Automatic differentiation 
in machine learning: a survey, J. Mach. Learn. Res. 18 (2018) 1 – 43 . 
[24] D. Voytan, M. K. Sen, Wave propagation with physics informed neural networks, 
in: SEG International Exposition and Annual Meeting, 2020. 
[25] S. Karimpouli, P. Tahmasebi, Physics informed machine learning: Seismic wave 
equation, Geosci. Front. 11 (6) (2020) 1993 – 2001 . 
[26] Z. Mao, A.D. Jagtap, G.E. Karniadakis, Physics-informed neural networks for high- 
speed flows, Comput. Meth. Appl. Mech. Eng. 360 (2020), 112789 . 
[27] M. Rautela, J. Senthilnath, J. Moll, S. Gopalakrishnan, Combined two-level damage 
identification strategy using ultrasonic guided waves and physical knowledge 
assisted machine learning, Ultrasonics 115 (2021), 106451 . 
[28] R. L. van Herten, A. Chiribiri, M. Breeuwer, M. Veta, C. M Scannell, Physics- 
informed neural networks for myocardial perfusion MRI quantification, arXiv 
preprint arXiv:2011.12844, 2020. 
[29] Z. Chen, Y. Liu, H. Sun, Physics-informed learning of governing equations from 
scarce data, Nat. Commun. 12 (1) (2021) 6136 . 
[30] M. Raissi, A. Yazdani, G.E. Karniadakis, Hidden fluid mechanics: Learning velocity 
and pressure fields from flow visualizations, Science 367 (6481) (2020) 
1026 – 1030 . 
[31] L. Lu, X. Meng, Z. Mao, G.E. Karniadakis, DeepXDE: A deep learning library for 
solving differential equations, SIAM Rev. 63 (1) (2021) 208 – 228 . 
[32] A. Pulkkinen, B. Werner, E. Martin, K. Hynynen, Numerical simulations of clinical 
focused ultrasound functional neurosurgery, Phys. Med. Biol. 59 (7) (2014) 
1679 – 1700 . 
[33] P.J. White, G.T. Clement, K. Hynynen, Longitudinal and shear mode ultrasound 
propagation in human skull bone, Ultrasound Med. Biol. 32 (7) (2006) 1085 – 1096 . 
[34] J. Robertson, E. Martin, B.T. Cox, B.E. Treeby, Sensitivity of simulated transcranial 
ultrasound fields to acoustic medium property maps, Phys. Med. Biol. 62 (7) 
(2017) 2559 – 2580 . 
[35] F. Marquet, M. Pernot, J.-F. Aubry, G. Montaldo, L. Marsac, M. Tanter, M. Fink, 
Non-invasive transcranial ultrasound therapy based on a 3D CT scan: protocol 
validation and in vitro results, Phys. Med. Biol. 54 (9) (2009) 2597 – 2613 . 
[36] P.B. Rosnitskiy, P.V. Yuldashev, O.A. Sapozhnikov, L.R. Gavrilov, V.A. Khokhlova, 
Simulation of nonlinear trans-skull focusing and formation of shocks in brain using 
a fully populated ultrasound array with aberration correction, J. Acoust. Soc. Am. 
146 (3) (2019) 1786 – 1798 . 
[37] J.-F. Aubry, M. Tanter, M. Pernot, J.-L. Thomas, M. Fink, Experimental 
demonstration of noninvasive transskull adaptive focusing based on prior 
computed tomography scans, J. Acoust. Soc. Am. 113 (1) (2003) 84 – 93 . 
[38] B.E. Treeby, J. Jaros, A.P. Rendell, B.T. Cox, Modeling nonlinear ultrasound 
propagation in heterogeneous media with power law absorption using a k-space 
pseudospectral method, J. Acoust. Soc. Am. 131 (6) (2012) 4324 – 4336 . 
[39] H. Igel, Computational seismology: a practical introduction, Oxford University 
Press, 2017 . 
[40] D. P. Kingma, J. Ba, Adam: A method for stochastic optimization, arXiv preprint 
arXiv:1412.6980, 2015. 
[41] Q. He, D. Barajas-Solano, G. Tartakovsky, A.M. Tartakovsky, Physics-informed 
neural networks for multiphysics data assimilation with application to subsurface 
transport, Adv. Water Resour. 141 (2020) . 
[42] B.E. Treeby, B.T. Cox, Modeling power law absorption and dispersion in 
viscoelastic solids using a split-field and the fractional Laplacian, J. Acoust. Soc. 
Am. 136 (4) (2014) 1499 – 1510 . 
[43] M.I. Iacono, E. Neufeld, E. Akinnagbe, K. Bower, J. Wolf, I. Vogiatzis Oikonomidis, 
D. Sharma, B. Lloyd, B.J. Wilm, M. Wyss, K.P. Pruessmann, A. Jakab, N. Makris, E. 
D. Cohen, N. Kuster, W. Kainz, L.M. Angelone, MIDA: A multimodal imaging-based 
detailed anatomical model of the human head and neck, PLoS One 10 (4) (2015) 
e0124126 . 
[44] L. Guasch, 
`
O. Calder ´on Agudo, M.-X. Tang, P. Nachev, M. Warner, Full-waveform 
inversion imaging of the human brain, npj Digit, Med. 3 (1) (2020) 1 – 12 . 
[45] C.-C. Huang, Y.-H. Lin, T.-Y. Liu, P.-Y. Lee, S.-H. Wang, Review: Study of the blood 
coagulation by ultrasound, J. Med. Biol. Eng. 31 (2) (2011) 79 – 86 . 
[46] T. Alkhalifah, C. Song, U. bin Waheed, Q. Hao, Wavefield solutions from machine 
learned functions constrained by the Helmholtz equation, Artif Intell Rev. 2 (2021) 
11-19. 
[47] V. Sitzmann, J. Martel, A. Bergman, D. Lindell, G. Wetzstein, Implicit neural 
representations with periodic activation functions, Adv. Neural Inf. Process. Syst. 
33 (2020) 7462 – 7473 . 
[48] C. Anitescu, E. Atroshchenko, N. Alajlan, T. Rabczuk, Artificial neural network 
methods for the solution of second order boundary value problems, Comput. 
Mater. Contin. 59 (1) (2019) 345 – 359 . 
[49] X. Wang, J. Li, D. Wang, et al., Sparse ultrasonic guided wave imaging with 
compressive sensing and deep learning[J], Mech. Syst. Signal Process. 178 (2022) 
109346 . 
[50] J. Lin, J. Li, C. Jiang, et al., Theoretical and experimental investigation of 
circumferential guided waves in orthotropic annuli[J], Ultrasonics 123 (2022) 
106715 . 
L. Wang et al.                                                                                                                                                                                                                                   