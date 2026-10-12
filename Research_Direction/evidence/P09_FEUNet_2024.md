# P09_FEUNet_2024.pdf

## PDF页 1

Finite difference-embedded UNet for solving transcranial
ultrasound frequency-domain wavefield
Linfeng Wang,1 Jian Li,1
 Shili Chen,1 Zheng Fan,2
 Zhoumo Zeng,1 and Y angLiu1,3,a)
1State Key Laboratory of Precision Measuring Technology and Instruments, Tianjin University, Tianjin, 300072, China
2School of Mechanical and Aerospace Engineering, Nanyang Technological University, 50 Nanyang Avenue,
Singapore 639798, Singapore
3International Institute for Innovative Design and Intelligent Manufacturing of Tianjin University in Zhejiang, Shaoxing 330100, China
ABSTRACT:
Transcranial ultrasound imaging assumes a growing signiﬁcance in the detection and monitoring of intracranial
lesions and cerebral blood ﬂow. Accurate solution of partial differential equation (PDE) is one of the prerequisites
for obtaining transcranial ultrasound waveﬁelds. Grid-based numerical solvers such as ﬁnite difference (FD) and
ﬁnite element methods have limitations including high computational costs and discretization errors. Purely data-
driven methods have relatively high demands on training datasets. The fact that physics-informed neural network
can only target the same model limits its application. In addition, compared to time-domain approaches, frequency-
domain solutions offer advantages of reducing computational complexity and enabling stable and accurate inver-
sions. Therefore, we introduce a framework called FD-embedded UNet (FEUNet) for solving frequency-domain
transcranial ultrasound waveﬁelds. The PDE error is calculated using the optimal 9-point FD operator, and it is inte-
grated with the data-driven error to jointly guide the network iterations. We showcase the effectiveness of this
approach through experiments involving idealized skull and brain models. FEUNet demonstrates versatility in han-
dling various input scenarios and excels in enhancing prediction accuracy, especially with limited datasets and noisy
information. Finally, we provide an overview of the advantages, limitations, and potential avenues for future
research in this study.
VC 2024 Acoustical Society of America . https://doi.org/10.1121/10.0025391
(Received 13 November 2023; revised 4 March 2024; accepted 8 March 2024; published online 27 March 2024)
[Editor: Guillaume Haiat] Pages: 2257–2269
I. INTRODUCTION
In contrast to medical imaging modalities like com-
puted tomography (CT) and magnetic resonance imaging
(MRI), transcranial ultrasound imaging offers the beneﬁts of
being non-invasive, portable, cost-effective, and highly
repeatable. It has unique advantages and values in detecting
and monitoring intracranial lesions and cerebral blood ﬂow.
Due to the fact that the infant skull has not yet fully devel-
oped, ultrasound imaging can directly observe the intracra-
nial situation through the fontanelle, making ultrasound
imaging a more promising method.
In recent years, there have been some excellent works
in the ﬁeld of solving transcranial waveﬁelds,
1–6 which
essentially involve solving partial differential equations
(PDEs). Aubry et al.
7 proposed a series of numerical bench-
marks related to transcranial ultrasound simulation, intro-
duced 11 modeling tools, and systematically compared the
numerical results obtained by different tools. The commonly
used grid-based numerical techniques, such as ﬁnite differ-
ence (FD) and ﬁnite element methods, have been effectively
utilized in the domain of medical ultrasound imaging.
8–11
However, due to the grid-based concept, the computational
complexity of these methods increases exponentially when
dealing with high-frequency or high-dimensional problems,
and even suffers from the curse of dimensionality.
12
Another important issue to consider is the error introduced
by discretization. Grid-based methods solve discretized
PDEs through time stepping. When the grid size is insufﬁ-
cient to meet the desired modeling resolution, discretization
errors may lead to inaccurate results.
13 In addition, the per-
formance of FD method is not satisfactory when facing with
irregular boundary problems. Moreover, the use of FD
method to solve anisotropic acoustic wave equations often
leads to shear wave artifacts linked to the Laplacian opera-
tor, particularly when sources are situated in the anisotropic
region.
14 Another aspect that cannot be ignored is that when
the physical parameters of the medium change slightly,
these numerical methods need to be iteratively restarted,
which brings some inconvenience to practical problem solv-
ing. Therefore, exploring an alternative approach to wave-
ﬁeld solving is of considerable importance.
Neural networks (NNs) and their variants, such as the
convolutional neural network (CNN) and recurrent neural
network, have always been regarded as universal tools for
solving various types of problems. Given a sufﬁcient
amount of training data and a NN architecture with enough
hidden layers and neurons, they can effectively grasp both
linear and non-linear connections between the input and
a)Email: ultrasonicslab@tju.edu.cn
J. Acoust. Soc. Am. 155 (3), March 2024 VC 2024 Acoustical Society of America 2257
ARTICLE...................................
 28 August 2024 04:14:44


## PDF页 2

output. Many studies have demonstrated that NNs can be
used as an alternative tool to obtain solutions to PDEs. 15–17
Despite achieving great success, there are still some limita-
tions with NNs. First, NNs perform poorly on small sample
datasets. NNs require a large amount of training data to
achieve excellent models in a vast parameter space. If the
training dataset is too small, it usually leads to overﬁtting or
underﬁtting. However, obtaining a large amount of data is
not always easy for practical problems. Second, there are
issues with handling imbalanced datasets. When the sample
data of one class is much greater than that of other classes,
NNs may tend to predict this class, leading to poor predic-
tion results for other classes. Third, there is a lower require-
ment for model interpretability. NNs are often viewed as
“black boxes” and their internal decision-making process is
difﬁcult to explain. Due to the lack of constraints in the
learning process from physical knowledge such as the con-
servation of energy and momentum, NNs can easily produce
solutions that deviate signiﬁcantly from the truth. In
response to the above situation, physics-informed neural
network (PINN) and its derivatives, as new paradigms for
solving PDEs, have attracted people’s attention and been
widely studied.
18–20 The core idea of PINN is to leverage
the automatic differentiation technique to encode the gov-
erning equations and boundary conditions, along with other
prior knowledge, within the error function of a NN. By
enforcing the physical laws as constraints, the network itera-
tions are guided and constrained according to the underlying
physics. The effectiveness of PINN in modeling wave equa-
tions has been demonstrated in both time and frequency-
domains.
4,21–24 However, PINN can only perform iterative
calculations on the same model each time. When we modify
any physical parameter in the model, PINN needs to be
retrained. PINN has higher requirements for the initial and
boundary conditions. To take advantage of it, more accurate
initial and boundary conditions need to be provided, other-
wise unreliable computational results will be obtained.
Furthermore, before applying PINN, it is necessary to refor-
mulate the PDE into a suitable form. Sometimes, the appli-
cation of PINN may be limited when dealing with complex
PDEs. Therefore, developing modeling techniques that are
easy to implement, rely less on datasets, and can be applied
to different models could potentially bring signiﬁcant bene-
ﬁts in the ﬁeld of transcranial ultrasound.
In addition to the above discussion, there is another
issue that cannot be ignored. It is computationally costly to
obtain the waveﬁeld solution in time domain, especially for
multiple sources.
25 Additionally, there is a signiﬁcant need
for memory to store the waveﬁeld information. Compared
with time-domain problems, solving the frequency-domain
waveﬁeld can markedly reduce computational costs for an
equivalent number of sources. Moreover, when using
frequency-domain data for inversion, it is only necessary to
choose a few key frequency points for ﬁtting in order to
obtain inversion results that are close to the actual model.
26
This can greatly reduce the amount of data and computa-
tional effort required. Furthermore, frequency-domain
inversion is generally more stable and accurate compared to
time-domain inversion, due to the weaker correlation
between different frequency components and the better
resistance to disturbance factors such as noise. 27,28 Through
the application of methods like full waveform inversion, the
solution of the extensive impedance matrix derived from the
wave equation (Helmholtz equation) yields Green’s func-
tions that can be utilized for model inversion. 14
In this paper, we introduce a framework called FD-
embedded UNet (FEUNet) to solve the frequency-domain
transcranial ultrasound waveﬁeld corresponding to the
Helmholtz equation. The proposed approach chooses UNet,
a widely used fully CNN consisting of an encoder and a
decoder, as the data-driven framework. The input of the net-
work is a two-dimensional (2D) brain velocity model, con-
taining clots of varying sizes and positions. The output is
the frequency-domain waveﬁeld corresponding to the
model, encompassing both the real and imaginary parts.
FEUNet is inspired by the idea of PINN and adds physical
constraints, the PDE error calculated from the governing
equation, to the data-driven error function. Both components
jointly constrain the network iterations, which is different
from purely data-driven methods. We acquire numerical
waveﬁeld solutions by employing an optimal 9-point FD
wave equation operator,
29 and use these results as the refer-
ence solutions to evaluate the performance of both the pure
data-driven method (UNet) and the proposed method
(FEUNet). We compared the performance of UNet and
FEUNet using the idealized skull model and 2D brain
model, respectively, and also discussed the impact of differ-
ent numbers of samples and signal-to-noise ratio (SNR) con-
ditions on the approaches. The results demonstrate that the
proposed method can accelerate the iteration process of the
network and improve the prediction performance, especially
for scenarios with limited samples and noisy data.
II. METHODOLOGY
A. Wave equation in frequency domain
Due to the intricate structure of the skull and its intri-
cate interaction with ultrasound waves, accurately modeling
the transmission of transcranial ultrasound presents a formi-
dable challenge. The coupling of solid and liquid compo-
nents, the presence of multiple layers, and the porous nature
of the skull all exert a signiﬁcant inﬂuence on the behavior
of ultrasound waves as they traverse it. Furthermore, the
skull’s high velocity relative to the internal tissues adds an
additional layer of complexity to the modeling process.
However, when we reasonably neglect the effects of shear
wave, nonlinearity, and acoustic absorption,
1,30–33 process
in frequency domain can be represented as
r2p x; xðÞ þ x2mp x; xðÞ ¼ s xs; xðÞ ; (1)
where p x; xðÞ is the complex pressure waveﬁeld, s xs; xðÞ
is the source function, x stands for the angular frequency,
x ¼ x; yfg represents the spatial coordinate in the Cartesian
2258 J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al.
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 3

coordinates, and xs represents the spatial location of the
acoustic source. We deﬁne m ¼ 1=v2 as the squared slowness
with v denoting the velocity and r2 ¼ @2=@x2 þ @2=@y2 as
the Laplacian operator. Equation (1) is alternatively labeled
as the Helmholtz equation. In numerical simulations, we
often use absorbing boundary conditions 34 or perfectly
matched layer (PML) techniques 35 to simulate inﬁnite space
without introducing interference from reﬂected waves, ensur-
ing the accuracy and stability of the calculations. Once PML
is incorporated into a 2D bounded rectangular area, the
Helmholtz equation can be represented by
25,36
@
@x M @p
@x
/C18/C19
þ @
@y N @p
@y
/C18/C19
þ Kx2mp x; xðÞ ¼ s xs; xðÞ ;
(2)
where the coefﬁcients M, N and K are respectively equal to
Py=Px, Px=Py; and PxPy. The values of Px and Py can be
determined by
Px;y ¼ 1 /C0 ifx;y
x : (3)
The purpose of introducing fx;y is to attenuate and absorb
wave energy within the boundary of PML. Beyond the
boundaries of PML, both f
x and fy have a value of zero.
Inside the PML, fx and fy are expressed as
fx;y ¼ 2pa0 f0
dx;y
DPML
/C18/C19 2
; (4)
where f0 represents the dominant frequency of the source
function and a0 is a constant coefﬁcient equal to 1.79. 36
DPML stands for the thickness of PML layers; dx and dy rep-
resent the horizontal and vertical distances from points
ðx; yÞ within the PML to the boundaries of the PML region,
respectively. By combining Eqs. (2)–(4), we can implement
the PML technique for waveﬁeld solution on a 2D brain
model.
B. The architecture of UNet
We chose UNet as the main framework for the NN to
solve the transcranial ultrasound frequency-domain wave-
ﬁeld. Its structure is shown in Fig. 1. UNet was originally
proposed by Ronneberger et al. in 2015 for medical image
segmentation37 and has since been applied to waveﬁeld sim-
ulation problems. In contrast to the original structure, we
changed the corresponding parameters such as the depth of
the network and the value of padding to make it more suit-
able for the speciﬁc problem addressed in this paper. The
input of the network is the 2D brain velocity model, includ-
ing blood clots of different sizes, positions, and velocities
(following Gaussian distribution). The output of the network
consists of two channels, representing the real and imagi-
nary components of the frequency-domain waveﬁeld corre-
sponding to the input model. The network architecture
consists of an encoder (referred to as the downsampling pro-
cess, shown in the upper part of Fig. 1) and a decoder
(referred to as the upsampling process, shown in the lower
part of Fig. 1), connected through multiple skip connections.
In the encoder, each downsampling step includes two 3 /C2 3
convolutional layers, followed by a Leaky ReLU activation
function and a 2 /C2 2 maxpooling layer. Within the decoder,
every upsampling stage consists of a 2 /C2 2 transposed con-
volutional layer, followed by a concatenation operation and
two consecutive 3 /C2 3 convolutional layers.
In NNs, the output q
i of the ith layer is calculated from
the output of the previous one,
qi ¼ f i Wiqi/C0 1 þ bi/C0/C1
; i ¼ 1; 2; …; L; (5)
where Wi is the weight matrix of the ith layer, bi is the
bias vector, qi/C0 1 represents the output of the ( i/C0 1)th layer,
FIG. 1. (Color online) The network
structure of UNet.
J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al. 2259
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 4

f i is the activation function. The training process of the
NN is an iterative process, with each iteration updating
and optimizing the weight matrix W and bias matrix b in
the network, so that the output of the network gradually
approaches the labeled value. The error function, in this
work, is the mean squared error (MSE), commonly used to
measure the error between the model’s output and the
actual label, with the optimization process striving to min-
imize it:
MSE
Data ¼ 1
ND
XND
i¼1
p /C0 p/C3ðÞ 2; (6)
where p and p/C3 represent the labeled solutions and the pre-
dicted values of the NN. MSEData here refers to the data-
driven MSE in UNet. ND is the number of collocation points
used to calculate MSEData. In all the works, this error term is
calculated using all grid points. The Adam optimizer is used
to minimize the error function.
38
C. The architecture of FD-embedded UNet (FEUNet)
Incorporating domain knowledge as constraints dur-
ing the model optimization process and implementing
these constraints by modifying the framework of the NNs
is a feasible way to improve network performance.
39
Based on this idea, algorithms such as PINN, TgNN (the-
ory-guided neural network)
40 have been proposed and suc-
cessfully applied in various ﬁelds. These algorithms share
almost identical computation principles, incorporating
governing equations as regularization terms into the total
error function of NNs, ensuring the predicted results are
close to physical mechanisms. These algorithms can only
target the same input each time, that is, each iteration opti-
mizes the waveﬁeld of the same model. In this paper, we
propose the FD-embedded UNet (FEUNet) framework,
and the algorithm framework is shown in Fig. 2. The big-
gest difference between the proposed algorithm and PINN
is that it uses the idea of adding governing equation con-
straints, but the input of the network is different models.
It combines the advantages of data-driven and physics-
driven modeling approaches. FEUNet can make predic-
tions on various different inputs and use physical
constraints to enhance the robustness and interpretab ility
of the prediction results. This may be very useful for many
practical problems. Compared to the input of PINN (coor-
dinates x, y,a n dt i m et for time-domain problems, coordi-
nates x, y, and angular frequency x for frequency-domain
problems), the input of FEUNet is a 2D brain model con-
taining blood clots with different sizes, positions, and
velocities. Therefore, automatic differentiation cannot be
used to obtain partial derivatives and calculate the PDE
error. In the numerical simulations of this paper, the train-
ing set is generated by an optimal 9-point FD wave equa-
tion operator. Therefore, we use the concept of it to
calculate the partial derivatives here.
Assume that an arbitrary 2D bounded domain is divided
into a uniform grid. Let p
x;y represents the pressure wave-
ﬁeld at coordinate ðx; yÞ and Dd stands for the node spacing
on the grid. According to the 9-point FD method, the
Laplacian operator can be expressed as
r
2px;y ¼ ar2
0ðÞ px;y þ 1 /C0 aðÞ r 45ðÞ
2px;y; (7)
where
r2
0ðÞ px;y ¼ pxþ1;y þ px/C0 1;y /C0 4px;y þ px;yþ1 þ px;y/C0 1
Dd2 ; (8)
r2
45ðÞ px;y ¼ pxþ1;yþ1 þ px/C0 1;yþ1 /C0 4px;y þ pxþ1;y/C0 1 þ px/C0 1;y/C0 1
2Dd2 :
(9)
The value of constant a is 0.5. Meanwhile, the FD approxi-
mation of the collocation point p of the mass acceleration
term x2mp in Eq. (1) can be represented as the following
linear combination:
px;y ¼ bpx;y þ cp xþ1;y þ px/C0 1;y þ px;yþ1 þ px;y/C0 1ðÞ
þ dp xþ1;yþ1 þ px/C0 1;yþ1 þ pxþ1;y/C0 1 þ px/C0 1;y/C0 1ðÞ ;
(10)
where b þ 4c þ 4d ¼ 1. The optimized coefﬁcients given
by Jo et al. 29 are b ¼ 0:6248, c ¼ 0:09381, and
d ¼/C0 0:00004. Substitution of Eqs. (7)–(10) into Eq. (1)
gives the following equation:
a pxþ1;y þ px/C0 1;y /C0 4px;y þ px;yþ1 þ px;y/C0 1
Dd2 þ 1 /C0 aðÞ
/C2 pxþ1;yþ1 þ px/C0 1;yþ1 /C0 4px;y þ pxþ1;y/C0 1 þ px/C0 1;y/C0 1
2Dd2
þ x2mb p x;y þ cp xþ1;y þ px/C0 1;y þ px;yþ1 þ px;y/C0 1ðÞ
/C2
þ 1 /C0 b /C0 4c
4 pxþ1;yþ1 þ px/C0 1;yþ1 þ pxþ1;y/C0 1ð
þ px/C0 1;y/C0 1
/C1/C3
¼ s: (11)
FIG. 2. (Color online) The workﬂow
diagram of FEUNet.
2260 J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al.
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 5

Equation (11) is the FD formulation of Eq. (1). By substitut-
ing the predicted frequency-domain waveﬁeld obtained
from the network into Eq. (11) and selecting N
P collocation
points for MSE calculation, we can obtain the physics-
driven error term MSE
Phy. The total error of the network is
obtained by weighted summation of data-driven error and
physics-driven error,
MSE
total ¼ kDataMSEData þ kPhyMSEPhy: (12)
The selection of weights for kData and kPhy is contingent on
the particular circumstances. One general solution is to
make each error term on the same order of magnitude. This
weight conﬁguration scheme can make the network easier to
converge. Yet in fact, there is no ﬁxed answer to the selec-
tion of the weights. Sometimes increasing or decreasing the
initial weights by a factor of ten may yield improved itera-
tive results. When the sample of the dataset decreases, it is
advisable to increase the weight parameters of physical con-
straint terms appropriately. We obtain the most appropriate
answer through trials and errors.
III. NUMERICAL SIMULATIONS
In this section, we will share the results of implement-
ing the proposed method on the idealized skulls and the 2D
brain model, respectively. Both the training and testing sets
were obtained from the optimal 9-point FD wave equation
operator. For all cases, the source function was implemented
using a Ricker wavelet. To partially alleviate the singularity
inherent in point sources, isotropic Gaussian sources with a
variance of 3 were employed as a surrogate.
25 During
frequency-domain waveﬁeld modeling, the Ricker wavelet
was converted from the time domain into a set of complex
values. The Gaussian source was multiplied by the appropri-
ate Fourier representation of the Ricker wavelet complex
value for a speciﬁc frequency. The batch size for all exam-
ples is set to 16.
Before conducting numerical simulations, it is neces-
sary to test the validity of the physical constraints by calcu-
lating the PDE errors before and after network training.
Here, we take the 2D brain model as an example (a detailed
introduction about this model will be provided later). As
shown in Fig. 12 in the Appendix, the PDE errors for 500
random points after network training are much smaller than
before. This indicates that the direction of the physical con-
straint is consistent with that of the network iteration, prov-
ing the effectiveness of the physical constraint. After the
completion of network training, we validate the prediction
performance of the network on the testing set. We use the
relative 2-norm error ‘
2 to characterize the magnitude of
prediction errors. The relative 2-norm error ‘ 2 is calculated
as follows:
‘ 2 ¼ kppre /C0 pref k2
kpref k2
: (13)
A. Performance comparison on idealized skull models
These examples are trained using a dataset of sound
speed distributions obtained from idealized models of cra-
nial structures. The training dataset was modiﬁed from the
original dataset mentioned in Ref. 1. Figure 3 displays sev-
eral velocity distribution examples derived from idealized
skull models. As the excitation source, we selected the
Ricker wavelet with a central frequency of 500 kHz, posi-
tioned at coordinates (131.25 mm, 248.75 mm). The ideal-
ized skull model is randomly generated and has a hollow
convex structure resembling an ellipse. Each example has a
constant thickness, but the thickness varies between different
examples, ranging from 5.5 mm to 27 mm. All the examples
have the same velocity of 2820 m/s, with a ﬁxed background
velocity of 1500 m/s. This conﬁguration ensures that the
sound speed contrast between the background and the ideal-
ized skull aligns with the speed contrast between soft tissue
and the human skull. Each example has a size of 1000 /C2 1000
grid points, with a normalized grid spacing of 0.25 mm. The
thickness of each boundary in PML is 25 grids. The training
set consists of 512 different idealized skull models, while the
testing set contains 100 distributions. This example is to dem-
onstrate the performance of the proposed method in
FIG. 3. Examples of the velocity distributions based on idealized skulls. The size of each case is 1000 /C2 1000, with a spacing of 0.25 mm.
J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al. 2261
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 6

processing skull models. 1,39 The number of sampling points
for calculating the PDE error is 2000, accounting for 0.2% of
the total number of the allocation points.
Training was conducted using both UNet and FEUNet
models, and the error curves are depicted in Fig. 4. From the
error curve, it can be observed that FEUNet achieves lower
training error (0.01822) compared to UNet (0.05260). In
addition, due to the incorporation of physical constraints,
FEUNet exhibits faster iteration speed, as evidenced by
requiring fewer iterations to reach the same error value. The
relative 2-norm errors ‘
2 for the 100 examples in the testing
set are shown in Fig. 5. Overall, the errors for FEUNet
model are consistently smaller than those for UNet.
Figure 6 presents four examples of waveﬁeld computa-
tions. For simplicity, we only compared the real part of the
waveﬁeld. From the predicted waveﬁeld, we can observe
that UNet is capable of predicting most of the waveﬁeld
information. We can also infer the approximate shape of the
input from the waveﬁeld. However, there is a signiﬁcant
error of waveﬁeld information after the ultrasound passes
through the idealized skull model. In contrast, with the
incorporation of physical constraints, FEUNet is better able
to predict the subtle waveﬁeld information, especially in the
regions near the skull where the waveﬁeld becomes more
complex due to sudden changes in velocity. The differences
between the references and the predictions provide a more
intuitive reﬂection of this result, where it is evident that the
FEUNet exhibits signiﬁcantly lower errors compared to
UNet. These results demonstrate that the physical con-
straints improve the predictive capability of the network,
making it easier for the network to capture the detailed
information of the waveﬁeld.
B. The impact of different numbers of observations
on predictions
These examples are trained based on the 2D brain
model. The 2D brain model is derived from the multimodal
imaging-based detailed anatomical (MIDA) model. This
model presents a highly detailed segmented cranial structure
through in-depth processing and analysis of high-quality
MRI, magnetic resonance angiography, and diffusion tensor
imaging scan data.
41 The MIDA model provides various tis-
sues and corresponding acoustic parameters such as velocity
and density for a 3D brain model. These parameters may not
be complete and missing values need to be retrieved from
literature. Cutting the MIDA model with a 0.5 mm interval
yields 465 slices, from which we selected one slice [as
shown in Fig. 7(a)] to build the 2D brain model [as shown in
Fig. 7(c)]. The prevailing consensus posits that the outer
structure of the brain model is immersed in a medium of
water, exhibiting a velocity of 1500 m/s. We can also obtain
the brain model parameters from CT or MRI images using
the relationship between Hounsﬁeld units (HU) and acoustic
parameters. This theory is widely recognized, and detailed
calculation formulas can be found in Refs. 29 and 30. Angla
et al.
42 provided a comprehensive summary of the calcula-
tion of brain acoustic parameters. We added blood clots
with different locations, sizes, and velocity distributions to
the original 2D brain model, and used them as the velocity
models in both the training and testing sets. These blood
clots are all circular in shape, with radii ranging from 8 to
20 mm. The centers of the clots are located within a circle
with a radius of 30 mm centered at the midpoint. The veloc-
ity distribution of blood clots adheres to a Gaussian distribu-
tion. We randomly generated a maximum value not
exceeding 1700 m/s and a minimum value not less than
1600 m/s. Subsequently, the velocity distribution between
the maximum and minimum values was generated following
a Gaussian distribution. Figure 7(d) shows one example of
the 2D brain model with a clot. The velocity proﬁle at the
black line is shown in Fig. 7(b). We do not have a ﬁxed
number of samples in the training set, as we will discuss the
inﬂuence of the number of samples on the proposed algo-
rithm. In the testing sets, we regenerated blood clots with
different positions, sizes, and velocity distributions. Unlike
the training sets, the velocity distribution of clots in the test-
ing set includes ﬁxed values of 1600 and 1700 m/s, as well
as velocities following a Gaussian distribution (as described
above). The size of each input in the examples is 151 /C2 151,
FIG. 4. (Color online) The network error of UNet and FEUNet for idealized
skull model.
FIG. 5. (Color online) The relative 2-norm error ‘ 2 of UNet and FEUNet
for idealized skull models in testing set.
2262 J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al.
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 7

and the output is 151 /C2 151 /C2 2, where “2” represents the
real and imaginary parts. The grid spacing is 2 mm.
Figure 7 depicts the 2D brain model (with and without a
blood clot) and their corresponding transcranial ultrasound
frequency-domain waveﬁelds. FEUNet has the ability to
convert governing equations into constraints and effectively
leverage domain knowledge to optimize network iterations.
In theory, it possesses greater learning capability and
requires less data compared to purely data-driven models
(UNet). To evaluate the effectiveness of the proposed
method, we compared the performance of FEUNet and
UNet on different numbers of training sets, ranging from 64
to 512. The excitation source with a main frequency of
300 kHz is located at (150 mm, 256 mm). The thickness of
each boundary in PML is 20 grids. For all the cases, a ran-
dom selection of 500 points (2% of the total allocation
points) are employed to calculate the PDE error.
Independent replication experiments were conducted to
avoid the inﬂuence of randomness on model evaluation.
After training, we obtained the predictions of the two meth-
ods on the same testing set and compared the results with a
reference solution obtained by the optimal 9-point FD calcu-
lations. Figure 8 shows the error curves and errors of
FEUNet and UNet under different numbers of training sets,
as well as their errors of the real part of frequency-domain
waveﬁeld compared to the reference solution. We can see
from the error curves that for different numbers of training
samples, the training errors of FEUNet are always smaller
than UNet. Additionally, FEUNet requires far fewer itera-
tions to reach the same error value as UNet, which reﬂects
FEUNet’s faster iteration speed compared to UNet. The
error curves for the real and imaginary parts in the PDE
error also demonstrate the iteration of the network.
Meanwhile, the errors of the waveﬁelds shows that
FEUNet has smaller errors than UNet, particularly in cases
where the training dataset is less extensive. This is quite
meaningful in practical problems, as a large amount of
training data is often difﬁcult to obtain in practical situa-
tions. The use of physical constraints can reduce the net-
work’s dependence on data and achieve better network
prediction results in the case of a small amount of data. It
should be noted that we adopt different weight conﬁgura-
tion schemes for different amounts of training data. The
initial solution i nvolves establishing a ratio between the
two weights, ensuring that the data-driven error and
physics-driven error have values within the same order of
magnitude. When the dataset is small, it is advisable to
increase the weight of the physical constraints (e.g., by a
factor of 10) appropriately. The speciﬁc weight conﬁgura-
tion scheme was obtained through trials and errors.
Furthermore, we calculated the relative 2-norm errors ‘
2
for both methods on 100 testing sets and drew a box plot
to compare the distribution of errors, as shown in Fig. 9.I t
is very clear that for all cases, the errors of FEUNet are
FIG. 6. (Color online) Four examples of simulations using idealized skull distributions randomly selected from the testing set. The idealized skull models
are covered on the waveﬁelds and the waveﬁelds have been normalized. In each case, the reference solution is obtained from the optimal 9-point FD wave
equation operator.
J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al. 2263
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 8

l o w e rt h a nt h a to fU N e t .W h e nt h en u m b e ro ft r a i n i n g
samples is 64, the error of UNet sharply increases to a
large value, indicating that the prediction of the waveﬁeld
by UNet has almost completely failed. The increase in
FEUNet error is relatively small, almost equal to the error
of UNet in the case of 128 samples, which proves that
physical constraints reduce the network’s dependence on
the training data. In addition, it can be seen from the con-
centration of errors that FEUNet has a smaller variance
value and better applicability to different inputs.
C. The impact of different SNR conditions
on predictions
Within this section, we explore the inﬂuence of noise on
the effectiveness of the proposed algorithm. Random noise,
following a normal distribution with a mean of zero, was
introduced to the complex waveﬁelds in the training set of the
2D brain model. The training set consists of 256 samples,
while the testing set contains 100 samples. The signal-to-noise
ratio (SNR) ranges from 10 to –10 dB. For each SNR condi-
tion, both algorithms were trained multiple times and the best
results on the testing set were selected for comparison. The
comparison of relative 2-norm error ‘
2 of UNet and FEUNet
under different SNR conditions is shown in Fig. 10.A st h e
SNR decreases, the errors of both methods gradually increase.
However, the error values of FEUNet are always lower than
those of UNet, and the upward trend is also smaller. It is more
evident that the performance of the network is improved with
the addition of physical constraints under low SNR conditions.
In the real world, observations inevitably contain both noise
and genuine values. Purely data-driven models struggle to dif-
ferentiate between authentic data and noise, making them sus-
ceptible to substantial interference. However, because true
values themselves conform to physical mechanisms, the incor-
poration of domain-speciﬁc knowledge empowers FEUNet to
effectively discriminate data noise, thereby enhancing the
model’s resilience to noise interference to a certain extent.
IV . DISCUSSION
A. Advantages of the proposed method
FEUNet is a NN-based forward encoder-decoder model.
It draws inspiration from PINN and incorporates governing
equations into the error function to constrain network itera-
tions and enhance network representation capabilities. This
improves the network’s expressive power and generalization
ability, allowing it to predict subtle acoustic information,
FIG. 7. (Color online) The 2D brain
model (with and without a blood clot)
and their corresponding transcranial
ultrasound frequency-domain wave-
ﬁelds. (a) A slice from MIDA; (c) 2D
brain model transformed from (a); (d)
2D brain model with a clot; (b) the
velocity proﬁle at the blood clot in (d);
(e) the frequency-domain waveﬁeld of
(c); and (f) the frequency-domain
waveﬁeld of (d). The red dashed lines
represent the PML. The waveﬁelds
have been normalized.
2264 J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al.
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 9

reducing the network’s dependence on the dataset and improv-
ing its noise resistance. It also retains the framework of tradi-
tional NNs, allowing it to be applicable to different input
models, which can be more challenging for PINN. Therefore,
it combines the advantages of multi-model training and
improving network performance. It is important to note that
when using the PINN framework for frequency-domain
modeling of the 2D brain model in this paper, it is necessary
to preprocess the velocity model by applying smoothing tech-
niques. This is because PINN may encounter difﬁculties when
dealing with boundaries with abrupt velocity changes. Here,
we use the method proposed in Ref. 25 (PINN for solving
frequency-domain waveﬁelds) for comparison and explana-
tion. The input for PINN is the 2D brain model with a size of
151 /C2 151 and a grid spacing of 2 mm. The output is the corre-
sponding frequency-domain waveﬁeld, which consists of both
the real and imaginary parts. As mentioned previously, the
optimal 9-point FD wave equation operator is employed to
generate the reference solution. We select the Ricker wavelet
with a frequency of 150 kHz as the excitation source, located
at (150 mm, 256 mm). Figure11 displays the prediction results
of PINN. For simplicity, only the real part of the waveﬁeld is
presented here. Figures 11(b) and 11(c) s h o wt h er e a lp a r to f
the waveﬁeld predicted by PINN for the original model, as
well as the difference compared to the reference solution. It is
evident that PINN has failed to predict the corresponding
waveﬁeld, particularly the information surrounding the brain,
FIG. 8. (Color online) The network error of UNet and FEUNet under different numbers of training sets, as well as the differences between their respecti ve
predicted solutions and the reference solutions. The reference solution is obtained from the optimal 9-point FD wave equation operator. The skull mo dels
are covered on the waveﬁelds and the waveﬁelds have been normalized.
FIG. 9. (Color online) The relative 2-norm error ‘ 2 of UNet and FEUNet
under different numbers of training sets.
J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al. 2265
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 10

which is almost negligible. After the original model is
smoothed, as shown in Fig.11(d), PINN is capable of accu-
rately obtaining the solution. This is also supported by the dif-
ference shown in Fig.11(f). By comparing the two sets of
results, we can infer that PINN requires prior smoothing pre-
processing of the velocity model when predicting the
frequency-domain waveﬁeld. In this example, we have chosen
a lower frequency of 150 kHz. The reason for selecting a lower
frequency is that PINN struggles to handle high-frequency
problems effectively. Increasing the frequency requires a
deeper network with a larger number of neurons. Despite the
low frequency, this example is sufﬁcient to illustrate the limita-
tions of PINN in predicting frequency-domain waveﬁelds and
its inability to handle models with signiﬁcant velocity ﬂuctua-
tions. In contrast, FEUNet is not constrained by the velocity
model. In addition, we compared the time required for different
methods (FEUNet, UNet, and FD) to obtain waveﬁelds for the
same testing set (sample size of 100), as shown in Table I in
the Appendix. The reason for the difference in the runtime of
the two models is the disparity in their sizes. The data size of
the idealized skull model is 210 /C2 210, while the 2D brain
model is sized at 150 /C2 150. This also indicates, from a side
perspective, that the increase in model size has a signiﬁcant
impact on the runtime of the FD method.
B. Limitations, solutions, and future directions
FEUNet achieves the proximity of prediction results to
the underlying physical laws by incorporating the governing
equations into the network’s error function as regularization
terms. This approach serves as a soft constraint, assessing
the weighted average consistency between the model predic-
tions and the physical laws. Although the regularization
terms constrain the network’s iterations to some extent and
ensure that the prediction results comply with the physical
laws on average, they cannot guarantee strict adherence to
the physical laws at every single point (which is a limitation
of the penalty method). In practice, it is possible to encounter
notably unrealistic physical predictions that signiﬁcantly
deviate from the governing equations. In contrast to soft con-
straints, hard constraints refer to projecting the predictions of
the NN onto a hyperplane constructed by governing equa-
tions, ensuring that the results on each constraint patch
strictly adhere to the physical constraints.
39,43 As a result,
hard constraints are more efﬁcacious than soft constraints
and can further enhance network performance. In theory, by
applying hard constraints, the model can achieve higher pre-
diction accuracy and stronger noise resistance while requir-
ing much less data. However, the implementation of hard
constraints is always more challenging than soft constraints.
Currently, the research on incorporating hard constraints into
deep learning is still at its early stages. Employing the imple-
mentation of hard constraints to improve the performance of
FEUNet would be a promising avenue for the future work.
In this study, we utilize a 300 kHz central frequency
signal to stimulate the waveﬁeld of the 2D brain model.
While this frequency is relatively low in comparison, opting
for a higher frequency such as 500 kHz, the FD method
would require the grid to be divided into multiple times in
order to obtain a dispersion-free waveﬁeld. However, in this
case, the number of points contained in the frequency-
domain waveﬁeld increases exponentially. Training a net-
work on this dataset would require a larger number of
parameters, which would be very time-consuming and may
even exceed the computational capabilities available at pre-
sent. In fact, FEUNet can be relatively easily extended to
handle problems involving other frequencies and models.
When dealing with different problems, we need to modify
the expression of the corresponding physical constraints
using the concept of FD method (as the governing equation
may be different), as well as the weight parameters.
Due to the embedding of physical constraints, additional
PDE error calculation is performed during network training,
which unavoidably increases the computational cost.
However, only 2.2% of the collocation points are added for the
calculation of PDE errors. More importantly, as compensation,
FEUNet offers higher prediction accuracy compared to UNet.
It is capable of capturing more detailed waveﬁeld information,
reducing reliance on the dataset, and enhancing its ability to
resist noise interference. In addition, from the error curve we
can observe that FEUNet has a faster iteration speed compared
FIG. 10. (Color online) The relative 2-
norm error ‘ 2 of UNet and FEUNet
under different SNR conditions.
2266 J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al.
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 11

to UNet. It requires fewer iteration steps to reach the same error
value. Compared to PINN, FEUNet exhibits better generaliza-
tion, as it can be applied to different models once the network
is well trained, without limitations on the velocity model. For
complex physical problems, the modeling process often neces-
sitates considering multiple constraint conditions to approxi-
mate the behavior of real systems. In such situations, the
design of the error function and the adjustment of its weights
become crucial to ensure the accurate ﬁtting of data and com-
pliance with physical constraints. However, in the face of these
challenges, the network architecture we proposed shares the
same training mechanism, which allows us to avoid cumber-
some computations of large matrix inversions. This method
provides an efﬁcient and feasible solution for solving complex
physical problems. By judiciously selecting constraint terms
and error function weights, we can achieve precise control
over the system’s behavior, leading to more reliable and accu-
rate simulation results. It is important to emphasize that this
method is not limited to speciﬁc ﬁelds of physics. Its generality
and scalability pave the way for in-depth exploration in physi-
cal modeling and offer new possibilities for interdisciplinary
studies, including ﬁelds like computer vision, natural language
processing, and biomedical research. Drawing inspiration from
this network architecture’s principles, we can tackle complex
real-world problems in these domains. Similarly, Chatillon
et al.
44 suggested an alternative approach by substituting the
computationally expensive direct solver with a surrogate
model constructed from a database of simulation results. This
method facilitates rapid and precise simulation of the 3D pres-
sure ﬁeld and thermal damages caused by HIFU in near real-
time. In addition, Granados et al.
45 proposed a physics-based,
explainable, fast, and reliable meta-model aiming at providing
a controllable data generation tool. Our future work will con-
sider learning and utilizing this model to obtain more accurate
solutions for multiple physical ﬁelds. Overall, this approach
presents a promising direction for both physical modeling and
interdisciplinary research. Its potential applications hold value
in pushing the boundaries of scientiﬁc knowledge and techno-
logical advancements.
Finally, this approach can be extended to 3D and more
complex waveﬁelds to obtain more accurate transcranial ultra-
sound waveﬁelds. It requires inputting 3D labeled data and
incorporating constraints from the 3D wave equation when
applying this approach. For more complex scenarios, such as
considering solid-liquid coupling and the porous nature that
closely resembles the actual conditions of the brain, as well as
accounting for attenuation and absorption effects during wave
propagation, we would need more accurate governing equa-
tions and physical constraint error terms. Additionally, apply-
ing appropriate boundary conditions and initial conditions
would also enhance the accuracy of the network in predicting
the waveﬁeld. It is also a valuable recommendation to achieve
adaptive conﬁguration of hyperparameters under soft con-
straint conditions. The combination of the data-driven frame-
work and the incorporation of physical constraints makes it
easier for this method to learn complex waveﬁeld information.
The implementation of this approach in transcranial imaging
applications suggests its potential applicability to other com-
plex physical systems as well.
V . CONCLUSION
In this paper, we propose an FD-embedded UNet
(FEUNet) framework to solve transcranial ultrasound
frequency-domain waveﬁelds. This method is based on a data-
driven framework while incorporating the concept of PINNs,
which embeds the governing equations into the error function
of the NN. Unlike PINNs, which take coordinates as inputs to
the network, this method takes different velocity models as
inputs and produces corresponding frequency-domain wave-
ﬁelds as outputs. It possesses the advantages of both data-
driven and physical constraint approaches. It can be applied to
FIG. 11. (Color online) The prediction results of PINN before and after smoothing. The red dashed lines represent the PML. The waveﬁelds have been
normalized.
J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al. 2267
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 12

different models while ensuring that the learned mapping rela-
tionship incorporates the underlying physical laws. The perfor-
mance of this method has been validated using the idealized
skull models and 2D brain model with blood clots. The results
demonstrate that it can achieve higher prediction accuracy and
capture ﬁner details of the waveﬁelds compared to UNet. The
embedding of physical constraints reduces the network’s reli-
ance on data and enhances its robustness against noise. Future
works include applying this approach to 3D brain imaging,
incorporating additional physical constraints such as boundary
conditions and initial conditions to improve network perfor-
mance, adopting self-adjusting penalty weights, applying the
concept of hard constraints and considering complex viscoelas-
tic anisotropic media or ﬂuid-saturated porous media. Before
applying it to imaging, further works are needed to extend the
method to more general model conﬁgurations, such as scenar-
ios with multiple excitation sources.
ACKNOWLEDGMENTS
This work was supported in part by the National
Science Foundation of China under Contract No. 61773283,
and in part by National Key R&D Program of China under
Contract No. 2018YFC0808600.
AUTHOR DECLARATIONS
Conflict of Interest
The authors declare that they have no known competing
ﬁnancial interests or personal relationships that could have
appeared to inﬂuence the work reported in this paper.
DATA AVAILABILITY
The data that support the ﬁndings of this study are
openly available in JASACode_FEUNet at https://ﬁgshare.
com/s/78ddfcc45fbbef8ee5dc.
APPENDIX
The PDE error before and after training for 500 random
points is shown in Fig. 12.
The comparison of the time required for obtain wave-
ﬁelds for the same testing set (sample size of 100) using dif-
ferent methods is shown in Table I.
1A. Stanziola, S. R. Arridge, B. T. Cox, and B. E. Treeby, “A Helmholtz
equation solver using unsupervised learning: Application to transcranial
ultrasound,” J. Comput. Phys. 441, 110430 (2021).
2P. Marty, C. Boehm, and A. Fichtner, “Acoustoelastic full-waveform
inversion for transcranial ultrasound computed tomography,” in Medical
Imaging 2021: Ultrasonic Imaging and Tomography (SPIE, Bellingham,
WA, 2021), Vol. 11602, pp. 210–229.
3E. P. Vienneau and B. C. Byram, “A coded excitation framework for high
SNR transcranial ultrasound imaging,” IEEE Trans. Med. Imaging 42,
2886–2898 (2023).
4L. Wang, H. Wang, L. Liang, J. Li, Z. Zeng, and Y. Liu, “Physics-
informed neural networks for transcranial ultrasound wave propagation,”
Ultrasonics 132, 107026 (2023).
5R. A. Drainville, S. Chatillon, D. Moore, J. Snell, F. Padilla, and C.
Lafon, “A simulation study on the sensitivity of transcranial ray-tracing
ultrasound modeling to skull properties,” J. Acoust. Soc. Am. 154,
1211–1225 (2023).
6T. Bancel, A. Houdouin, P. Annic, I. Rachmilevitch, Y. Shapira, M.
Tanter, and J. F. Aubry, “Comparison between ray-tracing and full-wave
simulation for transcranial ultrasound focusing on a clinical system using
the transfer matrix formalism,” IEEE Trans. Ultrason. Ferroelectr. Freq.
Control 68, 2554–2565 (2021).
7J. F. Aubry, O. bates, C. Boehm, K. B. Pauly, D. Christensen, C. Cueto,
P. G /C19elat, L. Guasch, J. Jaros, Y. Jing, R. Jones, N. Li, P. Marty, H.
Montanaro, E. Neufeld, S. Pichardo, G. Pinton, A. Pulkkinen, A.
Stanziola, A. Thielscher, B. Treeby, and E. van ’t Wout, “Benchmark
problems for transcranial ultrasound simulation: Intercomparison of com-
pressional wave models,” J. Acoust. Soc. Am. 152, 1003–1019 (2022).
8T. Schaeken, L. Hoogerbrugge, and E. Verschuur, “A non-reﬂecting wave
equation through directional wave-ﬁeld suppression and its ﬁnite differ-
ence implementation,” Sci. Rep. 12, 407 (2022).
9J. Ren, Ji. Li, C. Liu, S. Chen, L. Liang, and Y. Liu, “Deep learning with
physics-embedded neural network for full waveform ultrasonic brain
imaging,” IEEE Trans. Med. Imaging. (published online 2024).
10J. Tong, X. Wang, J. Ren, M. Lin, J. Li, H. Sun, F. Yin, L. Liang, and Y.
Liu, “Transcranial ultrasound imaging with decomposition descent
learning-based full waveform inversion,” IEEE Trans. Ultrason,
Ferroelectr. Freq. Control 69, 3297–3307 (2022).
11C. F. Otesteanu, S. J. Sanabria, and O. Goksel, “Robust reconstruction of
elasticity using ultrasound imaging and multi-frequency excitations,”
IEEE Trans. Med. Imag. 37, 2502–2513 (2018).
12S. Alkhadhr, X. Liu, and M. Almekkawy, “Modeling of the forward wave
propagation using physics-informed neural networks,” in 2021 IEEE
International Ultrasonics Symposium (IUS) (IEEE, Piscataway, NJ,
2021), pp. 1–4.
13S. Alkhadhr and M. Almekkawy, “Wave equation modeling via physics-
informed neural networks: Models of soft and hard constraints for initial
and boundary conditions,” Sensors 23, 2792 (2023).
14C. Song, T. Alkhalifah, and U. B. Waheed, “Solving the frequency-
domain acoustic VTI wave equation using physics-informed neural
networks,” Geophys. J. Int. 225, 846–859 (2021).
15G. D. Weymouth, “Data-driven multi-grid solver for accelerated pressure
projection,” Comput. Fluids 246, 105620 (2022).
16T. Qin, K. Wu, and D. Xiu, “Data driven governing equations approxima-
tion using deep neural networks,” J. Comput. Phys. 395, 620–635 (2019).
17Z. Jiang, J. Jiang, Q. Yao, and G. Yang, “A neural network-based PDE
solving algorithm with high precision,” Sci. Rep. 13, 4479 (2023).
FIG. 12. (Color online) The PDE error before and after training for 500 ran-
dom points.
TABLE I. Comparison of time required for different methods (unit:
seconds).
FEUNet UNet FD
Idealized skull model 17.8807 18.2621 74.2894
2D brain model 11.0852 11.1698 38.9433
2268 J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al.
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44


## PDF页 13

18M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural
networks: A deep learning framework for solving forward and inverse
problems involving nonlinear partial differential equations,” J. Comput.
Phys. 378, 686–707 (2019).
19L. Yuan, Y. Q. Ni, X. Y. Deng, and S. Hao, “A-PINN: Auxiliary physics
informed neural networks for forward and inverse problems of nonlinear
integro-differential equations,” J. Comput. Phys. 462, 111260 (2022).
20L. Lu, R. Pestourie, W. Yao, Z. Wang, F. Verdugo, and S. G. Johnson,
“Physics-informed neural networks with hard constraints for inverse
design,” SIAM J. Sci. Comput. 43, B1105–B1132 (2021).
21M. Rasht-Behesht, C. Huber, K. Shukla, and G. E. Karniadakis, “Physics-
informed neural networks (PINNs) for wave propagation and full wave-
form inversions,” JGR Solid Earth. 127, e2021JB023120 (2022).
22H. Wang, J. Li, L. Wang, L. Liang, Z. Zeng, and Y. Liu, “On acoustic
ﬁelds of complex scatters based on physics-informed neural networks,”
Ultrasonics 128, 106872 (2023).
23C. Song and T. Alkhalifah, “Waveﬁeld reconstruction inversion via
physics-informed neural networks,” IEEE Trans. Geosci. Remote Sens.
60, 5908012 (2022).
24R. Liang, W. Liu, L. Xu, X. Qu, and S. Kaewunruen, “Solving elastody-
namics via physics-informed neural network frequency domain method,”
Int. J. Mech. Sci 258, 108575 (2023).
25C. Song, T. Alkhalifah, and U. B. Waheed, “A versatile framework to
solve the Helmholtz equation using physics-informed neural networks,”
Geophys J. Int. 228, 1750–1762 (2022).
26X. Wang, M. Lin, J. Li, J. Tong, X. Huang, L. Liang, Z. Fan, and Y. Liu,
“Ultrasonic guided wave imaging with deep learning: Applications in cor-
rosion mapping,” Mech Syst Signal Process. 169, 108761 (2022).
27Y. Li, B. Han, L. M /C19etivier, and R. Brossier, “Optimal fourth-order stag-
gered-grid ﬁnite-difference scheme for 3D frequency-domain viscoelastic
wave modeling,” J. Comput. Phys. 321, 1055–1078 (2016).
28W. Cao, Q. Li, J. Zhang, and W. Zhang, “Accelerating 2D and 3D
frequency-domain seismic wave modeling through interpolating
frequency-domain waveﬁelds by deep learning,” Geophysics 87,
T315–T328 (2022).
29C. H. Jo, C. Shin, and J. H. Suh, “An optimal 9-point, ﬁnite-difference,
frequency-space, 2-D scalar wave extrapolator,” Geophysics 61, 529–537
(1996).
30J. Robertson, E. Martin, B. T. Cox, and B. E. Treeby, “Sensitivity of sim-
ulated transcranial ultrasound ﬁelds to acoustic medium property maps,”
Phys. Med. Biol. 62, 2559–2580 (2017).
31F. Marquet, M. Pernot, J.-F. Aubry, G. Montaldo, L. Marsac, M. Tanter,
and M. Fink, “Non-invasive transcranial ultrasound therapy based on a
3D CT scan: Protocol validation and in vitro results,” Phys. Med. Biol.
54, 2597–2613 (2009).
32P. B. Rosnitskiy, P. V. Yuldashev, O. A. Sapozhnikov, L. R. Gavrilov,
and V. A. Khokhlova, “Simulation of nonlinear trans-skull focusing and
formation of shocks in brain using a fully populated ultrasound array with
aberration correction,” J. Acoust. Soc. Am. 146, 1786–1798 (2019).
33B. E. Treeby, J. Jaros, A. P. Rendell, and B. T. Cox, “Modeling nonlinear
ultrasound propagation in heterogeneous media with power law absorp-
tion using a k-space pseudospectral method,” J. Acoust. Soc. Am. 131,
4324–4336 (2012).
34R. L. Higdon, “Absorbing boundary conditions for acoustic and elastic
waves in stratiﬁed media,” J. Comput. Phys. 101, 386–418 (1992).
35J. P. Berenger, “A perfectly matched layer for the absorption of electro-
magnetic waves,” J. Comput. Phys. 114, 185–200 (1994).
36Z. Chen, D. Cheng, W. Feng, and T. Wu, “An optimal 9-point ﬁnite dif-
ference scheme for the Helmholtz equation with Pml,” Int. J. Numer.
Anal Model. 10, 389–410 (2013).
37O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks
for biomedical image segmentation,” in Medical Image Computing and
Computer-Assisted Intervention–MICCAI 2015: 18th International
Conference, Munich, Germany, October 5–9 (Springer International
Publishing, New York, 2015), Pt. III 18, pp. 234–241.
38D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
arXiv:1412.6980 (2014).
39Y. Chen, D. Huang, D. Zhang, J. Zeng, N. Wang, H. Zhang, and J. Yan,
“Theory-guided hard constraint projection (HCP): A knowledge-based
data-driven scientiﬁc machine learning method,” J. Comput. Phys. 445,
110624 (2021).
40N. Wang, D. Zhang, H. Chang, and H. Li, “Deep learning of subsurface
ﬂow via theory-guided neural network,” J. Hydrol. 584, 124700 (2020).
41M. I. Iacono, E. Neufeld, E. Akinnagbe, K. Bower, J. Wolf, I. V.
Oikonomidis, D. Sharma, B. Lloyd, B. J. Wilm, M. Wyss, K. P.
Pruessmann, A. Jakab, N. Makris, E. D. Cohen, N. Kuster, W. Kainz, and
L. M. Angelone, “MIDA: A multimodal imaging-based detailed
anatomical model of the human head and neck,” PLoS One 10, e0124126
(2015).
42C. Angla, B. Larrat, J. Gennisson, and S. Chatillon, “Transcranial ultra-
sound simulations: A review,” Med. Phys. 50, 1051–1072 (2023).
43Y. Chen and D. Zhang, “Integration of knowledge and data in machine
learning,” arXiv:2202.10337 (2022).
44S. Chatillon, R. Loyet, L. Brunel, F. Chavrier, N. Guillen, and N. Guillen,
“Applications of intensive HIFU simulation based on surrogate models using
the CIVA HealthCare platform,”J. Phys: Conf. Ser.1761(1), 012007 (2021).
45G. E. Granados, R. Miorelli, F. Gatti, S. Robert, and D. Clouteau,
“Towards a multi-ﬁdelity deep learning framework for a fast and realistic
generation of ultrasonic images in complex geometries with the multi-
modal total focusing method,” NDT&E Int. 139, 102906 (2023).
J. Acoust. Soc. Am. 155 (3), March 2024 Wang et al. 2269
https://doi.org/10.1121/10.0025391
 28 August 2024 04:14:44
