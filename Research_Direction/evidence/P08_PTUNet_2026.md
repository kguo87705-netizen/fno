# P08_PTUNet_2026.pdf

## PDF页 1

S0 Lamb wave scattering in plate structures using 
physics-enhanced TransUNet
Linfeng Wang
a
, Hongyan Zhang
a
, Zhen Zhang
a
, He Sun
a
, Xiao Ying
a
, Jing Hou
b
, Jian Li
a
,  
Zhoumo Zeng
a
, Yang Liu
a , c , *
a
State Key Laboratory of Precision Measurement Technology and Instruments, Tianjin University, Tianjin 300072, China
b
CNOOC Research Institute Co., Ltd., Beijing 100028, China
c
International Institute for Innovative Design and Intelligent Manufacturing of Tianjin University in Zhejiang, Shaoxing 330100, China
ARTICLE INFO
Keywords:
Lamb wave scattering
Physical constrains
Irregular defects
Wave propagation
ABSTRACT
Lamb wave scattering offers valuable insights into material properties, defect characteristics, and wave propa -
gation behaviors, making it a prominent research topic in the fields of non-destructive testing and structural 
health monitoring. Analytical methods, while offering theoretical analysis, are often limited to simple defect 
geometries and homogeneous media. Numerical methods such as the boundary element method and hybrid 
techniques can handle complex structures but suffer from computational costs, particularly at high frequencies 
due to the need for fine discretization. To overcome these challenges, this study proposes a physics-enhanced 
TransUNet (PTUNet) for solving the scattered wavefield of S0 Lamb waves in plate structures with irregular 
defects. PTUNet combines the local feature extraction capability with the global modeling capacity while 
employing finite difference approach to incorporate the Kirchhoff-Love plate theory as a physical constraint. The 
effectiveness of the proposed method is validated through numerical simulations on random defects and 
experimental measurements using a scanning laser Doppler vibrometer on a 1.5 mm-thick aluminum plate. The 
results demonstrate that PTUNet accurately predicts wave propagation, mode conversion, and intricate scat -
tering characteristics, achieving reasonable agreement with both finite element simulations and experimental 
observations. PTUNet presents a promising approach for wavefield modeling and defect characterization, with 
potential applications in acoustic scattering problems.
1. Introduction
In guided wave propagation, scattering refers to the alteration of 
wave paths and characteristics when encountering structural disconti -
nuities, defects, or geometric variations. Scattering is both a funda -
mental wave phenomenon in physics and a critical area of research in 
nondestructive testing (NDT). The analysis of scattering waves provides 
valuable insights into structural defects, material properties, and stress 
distribution, facilitating effective structural health monitoring and 
evaluation. In comparison to traditional inspection techniques, guided 
wave scattering offers significant advantages in NDT, especially for large 
or complex structures. First, guided wave scattering provides global 
information with strong penetration capability, allowing for the effec -
tive detection of defects in large-area or thick-walled structur -
es — something that conventional localized inspection techniques often 
struggle to achieve [ 1 ]. Second, scattering exhibits high sensitivity to 
small defects, enabling the detection of hidden flaws that may not be 
identifiable with traditional methods [ 2 ]. Furthermore, analyzing vari -
ations in the scattering waveforms provides multidimensional insights 
into the structural state, enhancing defect localization accuracy and 
detection efficiency [ 3 ]. Finally, guided wave scattering demonstrates 
remarkable adaptability, making it applicable across a wide range of 
materials and structures [ 4 ]. This is particularly beneficial for inspecting 
complex geometries or areas that are difficult to access, where tradi -
tional methods may have limitations. Through a systematic study of 
guided wave scattering mechanisms, researchers can develop a more 
accurate theoretical foundation and practical guidelines for assessing 
the reliability of materials and structures, thus advancing the field of 
NDT.
Over the past few decades, extensive research has been conducted on 
the scattering of guided waves interacting with defects in plate struc -
tures. Fundamental investigations into guided waves in elastic solids can 
* Corresponding author at: State Key Laboratory of Precision Measuring Technology and Instruments, Tianjin University, Tianjin 300072, China.
E-mail address: ultrasonicslab@tju.edu.cn (Y. Liu). 
Contents lists available at ScienceDirect
Ultrasonics
journal homepag e: www.else vier.com/loc ate/ultras
https://doi.org/10.1016/j.ultras.2025.107852
Received 7 May 2025; Received in revised form 25 August 2025; Accepted 6 October 2025  
Ultrasonics 159 (2026) 107852 
Available online 8 October 2025 
0041-624X/© 2025 Elsevier B.V. All rights are reserved, including those for text and data mining, AI training, and similar technologies. 

## PDF页 2

be found in the works of Achenbach [ 5 ] and Auld [ 6 ]. A notable study by 
Diligent et al. [ 7 ] focused on the scattering of incident S0 Lamb waves 
from circular through-holes in plates. By employing 3D theory and 
expanding the Lamb and SH wave fields using Bessel functions, the 
authors applied a projection method incorporating stress and displace -
ment boundary conditions to determine the expansion coefficients. 
Simulations and experimental validation revealed that scattering co -
efficients increase with hole diameter and decrease with distance from 
the hole. Building on this work, Grahn [ 8 ] and Moreau et al. [ 9 ] 
extended the study of 3D scattering of S0 Lamb waves to partially 
through-thickness circular cavities and irregularly shaped blind holes. It 
is noteworthy that [ 7 ] and [ 8 ] selected sine functions as projection 
functions, which were found to be unstable with slow convergence rates. 
In contrast, [ 9 ] adopted displacement components of Lamb waves as 
projection functions, significantly improving convergence speed. While 
precise 3D theory remains effective across both low and high fre -
quencies, its computational complexity is a major limitation. To address 
this challenge, researchers have explored plate theories as an alternative 
for modeling low-frequency scattering. Specifically, Poisson theory and 
Kirchhoff theory have been employed to characterize the extensional 
and flexural motions of guided waves in plates. Grahn [ 8 ] formulated 
the scattering problem using Poisson-Kirchhoff theory, converting 
displacement, stress, and moment boundary conditions into matrix 
form, and solving for the unknown expansion coefficients. This 
approach allowed for the calculation of far-field amplitudes of scattered 
S0, SH0, and A0 waves resulting from S0 Lamb wave incidence on a 
circular blind hole. A comparison with 3D theory demonstrated that 
Poisson-Kirchhoff theory provides high accuracy at low-frequencies. 
However, as the incident frequency increases, its effectiveness di -
minishes. To overcome this limitation, Kirchhoff theory was replaced by 
Mindlin theory, which maintains similar computational efficiency while 
offering a more accurate representation of flexural wave propagation in 
plates. Cegla et al. [ 10 ] employed Poisson-Mindlin theory to model 
guided wave scattering at non-symmetric blind holes in isotropic plates. 
Comparisons with 3D analytical solutions and acoustic field scanning 
experiments confirmed that Mindlin theory outperforms Kirchhoff the -
ory in high-frequency scenarios. Further extending this approach, 
Sedaghati et al. [ 11 ] developed an S0 Lamb wave scattering model for 
circular cavities in isotropic plates, capable of handling both symmetric 
and asymmetric single- and double-sided blind holes. Similar studies can 
be found in [ 12 , 13 ]. In recent years, Poddar and Giurgiutiu [ 14 ] intro -
duced a novel analytical method, Complex Modes Expansion with Vec -
tor Projection (CMEP), for solving Lamb wave scattering problems in 
plates. Unlike previous projection methods, CMEP enhances accuracy by 
projecting stress boundary conditions onto the conjugate of displace -
ment modes and vice versa. This formulation leverages power flow 
associated with Lamb wave modes, transforming stress and displace -
ment equations into power equations, leading to faster convergence. 
This method was applied to analyze the scattering behavior of horizontal 
cracks, disbands [ 14 ], steps [ 15 , 16 ], and horizontal cracks in a stiffened 
plate [ 17 , 18 ]. Moreover, Shen and Giurgiutiu [ 19 ] developed WFR-2D, 
an analytical prediction tool based on precise 2D Lamb wave modeling, 
which facilitates ultrasonic guided wave simulation in damaged struc -
tures. Research has also extended beyond individual defects to explore 
guided wave scattering in plates with multiple defects, as demonstrated 
in [ 20 , 21 ]. Extensive analytical research has been conducted on scat -
tering problems in plates. However, the mathematical complexity of 
analytical methods has restricted most studies to simple defect geome -
tries, primarily focusing on the angular-dependent energy distribution 
characteristics of the scattered wavefield. As a result, current analytical 
approaches have yet to fully capture time-domain scattered signals or 
reconstruct complete scattered wavefields.
In addition to analytical solutions, numerical methods such as the 
finite element method (FEM), boundary element method (BEM), and 
spectral element method (SEM) have been widely used to study wave 
scattering [ 22–24 ]. These methods often offer advantages when dealing 
with complex scatterers, where analytical approaches may become 
intractable. To improve computational efficiency, especially in high- 
frequency scattering problems with complex defect geometries, re -
searchers have developed hybrid methods that combine FEM, BEM, and 
mode expansion. For instance, Cho et al. [ 25 ] employed a hybrid BEM 
approach that integrates boundary integral equations with mode 
expansion techniques to numerically determine reflection and trans -
mission coefficients for different defect conditions. However, numerical 
methods inevitably involve a trade-off between solution accuracy and 
computational cost, particularly regarding mesh resolution. This chal -
lenge intensifies in high-frequency problems, which demand finer 
meshes and significantly increase computational complexity. A practical 
alternative is the semi-analytical finite element method (SAFE), which 
allows for 3D scattering analysis using a 2D guided wave framework. For 
example, Maruyama et al. [ 26 ] employed a hybrid SAFE-BEM method to 
investigate wave scattering in isotropic, linear elastic plates with finite 
width. Despite their benefits, these methods still require complex 
analytical formulations within the FE framework, which poses chal -
lenges in terms of user-friendliness and practical applicability. Further 
improvements are necessary to improve accessibility and computational 
efficiency for broader applications in guided wave scattering analysis.
With advancements in deep learning, neural networks have become 
a powerful and versatile tool for wave scattering analysis. In recent 
years, researchers have increasingly applied deep learning to wave 
scattering, introducing innovative methodologies and improving 
computational efficiency. For example, Shao et al. [ 27 ] proposed a 
multi-task convolutional neural network to investigate Lamb wave 
scattering in composite plates. By incorporating the Born approximation 
principle to generate a multi-damage dataset, this approach enhanced 
the accuracy of damage quantification. Ji et al. [ 28 ] introduced an 
asymmetric autoencoder architecture for the full wavefield reconstruc -
tion of Lamb waves in carbon fiber-reinforced plastic (CFRP) structures, 
achieving high reconstruction accuracy even under high sparsity con -
ditions and exhibiting strong generalization across different damage 
scenarios. Beyond these studies, deep learning has been effectively 
applied to wave propagation and damage detection, showcasing its 
broad potential in this field [ 29–31 ]. Despite their success, data-driven 
deep learning models rely heavily on large, high-quality training data -
sets. The scarcity or imbalance of labeled data across different damage 
types can severely affect the generalization and predictive accuracy of 
these models. Theoretically, the required training data grows exponen -
tially with the input dimensionality, yet in real-world engineering ap -
plications, acquiring such extensive datasets is often impractical. This 
data limitation has become a major challenge, restricting the broader 
adoption of deep learning in practical wave scattering problems.
To overcome these limitations, physics-informed neural networks 
(PINNs) and their variants [ 32–35 ] have emerged as a promising alter -
native, attracting significant attention across various research domains. 
Unlike conventional data-driven models, PINNs incorporate funda -
mental physical laws such as governing equations and boundary con -
ditions directly into the loss function using automatic differentiation. 
This approach ensures that the model learns from data while strictly 
adhering to physical constraints, improving both accuracy and gener -
alization. Compared to numerical methods, PINNs eliminate the need for 
meshing, mitigating the trade-off between computational cost and so -
lution accuracy. This advantage makes them particularly effective for 
complex geometries and high-dimensional problems. Furthermore, 
PINNs require significantly fewer labeled datasets, as they can be trained 
solely based on governing equations, ensuring physically consistent 
predictions even in out-of-distribution scenarios. PINNs have already 
shown great promise in ultrasonic wave scattering applications. In 
previous work, we have successfully employed PINN frameworks to 
model ultrasonic wave propagation in plates containing single and 
multiple defects, validating their effectiveness in solving complex scat -
tering problems [ 36 ]. Sun et al. [ 37 ] utilized a high-frequency shear 
horizontal (SH) guided wave-based PINN for microcrack quantification, 
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
2 

## PDF页 3

reducing quantification error by over 80 % compared to conventional 
methods. Zhou et al. [ 38 ] applied PINNs to extract undamaged wave -
fields from damage-affected wavefields, enabling precise defect identi -
fication using signal processing techniques. Go et al. [ 39 ] employed 
PINNs to analyze elastoacoustic interactions in perforated and cracked 
plates, achieving millimeter-scale accuracy in stress and displacement 
estimations. These studies collectively highlight the effectiveness of 
PINNs in wavefield reconstruction, defect quantification, and damage 
localization. However, research leveraging this approach for Lamb wave 
scattered wavefield reconstruction remains notably scarce.
Unlike conventional neural networks which are typically designed to 
approximate finite-dimensional input – output mappings (i.e., point-to- 
point or vector-to-vector relations), neural operator methods aim to 
directly learn mappings between functions. Representative approaches 
include DeepONet and Fourier Neural Operator (FNO). DeepONet em -
ploys a branch – trunk architecture, where the branch encodes the 
sampled values of the input function and the trunk encodes the output 
coordinates, with their inner product approximating the operator 
[ 40 , 41 ]. However, for large-scale spatiotemporal wavefields, it requires 
explicit ( x , y , t ) grids and dense sampling, leading to high dimensionality 
and training cost. FNO, on the other hand, leverages Fourier transforms 
to perform global convolutions in the spectral domain, making it 
particularly powerful for capturing large-scale propagation [ 42 , 43 ]. Yet, 
being data-driven, FNO does not strictly enforce governing PDEs, which 
can reduce physical consistency in local scattering. Prior studies have 
also suggested that adding PDE constraints can improve generalization 
in more challenging settings [ 44 ].
This paper proposes a physics-enhanced TransUNet (PTUNet) 
framework for solving the scattered wavefields of S0 Lamb wave in plate 
structures by integrating data-driven learning with physical constraints. 
Previous research has demonstrated the effectiveness of incorporating 
physical priors into data-driven models to improve prediction accuracy 
and generalization ability [ 45 ]. We embed Kirchhoff-Love plate theory 
into the TransUNet framework using finite difference (FD) technique as 
a physical constraint within the loss function. This integration enables 
PTUNet to jointly leverage data-driven learning and fundamental 
physical laws. Compared to purely data-driven approaches, the incor -
poration of physical constraints significantly reduces the reliance on 
large-scale labeled datasets, prevents unphysical solutions, and en -
hances model generalization. It should be emphasized that the “ physics- 
enhanced ” paradigm is not intended to replace other methods, but 
rather as an integration of physical priors into deep models. In this 
study, we conduct FE simulations to model the propagation of S0 Lamb 
waves incident on irregular defects in 3D plate structures, constructing a 
dataset containing out-of-plane displacement information. Numerical 
simulations validate the effectiveness of the proposed method in 
improving prediction accuracy and enhancing adaptability for varying 
defect geometries. Furthermore, scanning laser Doppler vibrometer 
(SLDV) experiments are conducted to obtain high-resolution wavefield 
measurements of a 1.5 mm-thick aluminium plate containing an irreg -
ular defect. Experimental results demonstrate that the predictions of 
PTUNet achieve reasonable consistency with FE simulations and 
experimental measurements. These findings highlight PTUNet as a 
robust and efficient framework for guided wave scattering analysis, of -
fering a promising solution for NDT and structural health monitoring 
applications.
2. Methodology
2.1. Propagation of Lamb waves in isotropic and homogeneous plates
2.1.1. 3D theory
The propagation theory of Lamb modes and shear horizontally 
polarized (SH) modes in isotropic, homogeneous, linearly elastic, and 
infinitely extended plates has been discussed in detail in [ 8 , 46 ]. Fig. 1
depicts the dispersion curve of Lamb waves in a 1.5 mm aluminium 
plate. The low-frequency S0 Lamb wave, with high group velocity and 
low attenuation, effectively detects deep defects and experiences mini -
mal attenuation in fluid media, making it ideal for long-range NDT of 
plate structures. Although the lowest frequency is preferred to avoid 
dispersion, a balance between resolution and frequency is often neces -
sary in practice. The frequency-thickness product is typically limited to 
1.6 MHz ‧ mm to avoid the A1 mode cut-off frequency [ 7 ]. This study 
focuses on frequencies of 150 kHz ‧ mm, where the S0 mode exhibits 
negligible dispersion.
The material properties of the plate are characterized by Young ’ s 
modulus E and Poisson ’ s ratio ν , or alternatively by the Lam ´e constants λ 
and μ , with the density denoted as ρ . The thickness of the plate is 2 h . The 
following derivation is formulated under the standard assumptions of 
time-harmonic fields, linear elasticity, and plate isotropy. Assuming the 
incident wave is a plane far-field S0 wave and the time factor exp (   iwt )
is omitted under the time-harmonic condition. In a cylindrical coordi -
nate system, the wave motion of Lamb modes can be expressed as: 
u
n
r
=
1
k
n
V
n
( z )
∂ ϕ
∂ r
( r , θ ) , u
n
θ
=
1
k
n
V
n
( z )
∂ ϕ
∂ θ
( r , θ ) , u
n
z
= W
n
( z ) ϕ ( r , θ ) . (1) 
The expressions of V
n
( z ) and W
n
( z ) for symmetric and antisymmetric 
modes in Lamb waves are provided in [ 46 ]. The scalar potential function 
ϕ ( r , θ ) satisfies the Helmholtz equation: 
∂
2
ϕ
∂ r
2
+
1
r
∂ ϕ
∂ r
+
1
r
2
∂
2
ϕ
∂ θ
2
+ k
2
n
ϕ = 0 . (2) 
The wavenumbers k
n 
for symmetric and antisymmetric modes are 
given from the Lamb-Rayleigh equation: 
tan ( qh )
tan ( ph )
=  
4 k
2
pq
 
q
2
  k
2
)
2
, for symmetric modes ,
tan ( qh )
tan ( ph )
=  
 
q
2
  k
2
)
2
4 k
2
pq
, for antisymmetric modes ,
(3) 
where 
q
2
=
ω
2
c
2
T
  k
2
, p
2
=
ω
2
c
2
L
  k
2
. (4) 
The quantities c
T 
and c
L 
represent the velocities of shear waves and 
longitudinal bulk waves, respectively, while ω denotes the angular fre -
quency. Now, we assume that a solution for the potential function ϕ ( r , θ ) , 
satisfying Eq. (2) , can be expressed as: 
Fig. 1. Group velocity dispersion curve of Lamb waves.
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
3 

## PDF页 4

ϕ ( r , θ ) = Φ ( r ) e
im θ
, (5) 
where Φ ( r ) is a solution of Bessel ’ s differential equation and satisfies 
d
2
Φ
dr
2
+
1
r
d Φ
dr
+
(
k
2
n
 
m
2
r
2
)
Φ = 0 . (6) 
The wavefield is divided by the boundary of the defect into two re -
gions: the outer wavefield (denoted by “ ＞ ” ) and the inner wavefield 
(denoted by “ ＜ ” ). For outgoing waves of the Lamb modes, the solutions 
of Eq. (2) are: 
ϕ
>
mn
( r , θ ) = H
m
(
k
Sh
n
r
)
e
im θ
, for symmetric modes ,
ϕ
<
mn
( r , θ ) = H
m
(
k
Ah
n
r
)
e
im θ
, for antisymmetric modes ,
(7) 
where H
m 
is the first type of Hankel functions. The wavenumbers k
Sh
n 
and 
k
Ah
n 
are obtained from Eqs. (3) and (4) , representing symmetric and 
antisymmetric modes, respectively. The half plate thickness is h . Assume 
that the remaining thickness of the plate inside the defect is 2 b . Then in 
the inner region, the solutions of Eq. (2) are: 
ϕ
<
mn
( r , θ ) = J
m
(
k
Sb
n
r
)
e
im θ
, for symmetric modes ,
ϕ
<
mn
( r , θ ) = J
m
(
k
Ab
n
r
)
e
im θ
, for antisymmetric modes ,
(8) 
where J
m 
is the first type of Bessel functions.
The scattering process also gives rise to SH modes, which can like -
wise be expressed using a scalar potential function ψ ( r , θ ) and a 
thickness-dependent function U
n
( z ) : 
u
n
r
=
1
l
n
U
n
( z )
1
r
∂ψ
∂ θ
( r , θ ) , u
n
θ
=  
1
l
n
U
n
( z )
∂ψ
∂ r
( r , θ ) , u
n
z
= 0 , (9) 
where ψ ( r , θ ) also satisfies the Helmholtz equation, and l
n 
corresponds to 
the wavenumbers for the SH modes. The thickness-dependent function 
U
n
( z ) can be found in [ 46 ]. Similarly, the solutions for SH modes are 
given as: 
ψ
>
mn
( r , θ ) = H
m
(
l
h
n
r
)
e
im θ
, (10) 
and 
ψ
<
mn
( r , θ ) = J
m
(
l
b
n
r
)
e
im θ
. (11) 
As we have previously assumed a plane S0 wave incidence, the po -
tential function for the incident field is given by: 
ϕ
inc
= e
ik
Sh
0
x 1
= e
ik
Sh
0
r cos θ
=
∑
∞
m =  ∞
i
m
J
m
(
k
Sh
0
r
)
e
im θ
. (12) 
According to Eq. (1) , the displacement field of the incident wave can 
be further derived: 
u
inc
r
=
∑
∞
m =  ∞
i
m
V
Sh
0
( z ) J
ʹ
m
(
k
Sh
0
r
)
e
im θ
. (13) 
At a given frequency, Eq. (3) has numerous roots in the complex 
plane [ 15 ]. Retaining only the roots corresponding to forward propa -
gating or evanescent modes, the scattering displacement field in the 
outer region can be expressed as: 
u
>
r
=
∑
∞
m =  ∞
∑
∞
n = 0
{
a
mn
V
Sh
n
( z ) H
ʹ
m
(
k
Sh
n
r
)
+ b
mn
V
Ah
n
( z ) H
ʹ
m
(
k
Ah
n
r
)
+ imc
mn
U
h
n
( z )
H
m
(
l
h
n
r
)
l
h
n
r
}
e
im θ
. (14) 
Similarly, the displacement in the inner region is: 
u
<
r
=
∑
∞
m =  ∞
∑
∞
n = 0
{
d
mn
V
Sb
n
( z ) J
ʹ
m
(
k
Sb
n
r
)
+ e
mn
V
Ab
n
( z ) J
ʹ
m
(
k
Ab
n
r
)
+ imf
mn
U
b
n
( z )
J
m
(
l
b
n
r
)
l
b
n
r
}
e
im θ
. (15) 
The expansion coefficients a
mn
  f
mn 
need to be determined to obtain 
the scattered displacement fields. In addition to the radial displacement 
given above, other displacement components can also be derived 
through a similar procedure. Furthermore, by applying Hooke ’ s law, the 
relevant stress components can also be determined. Next, we will 
determine the expansion coefficients by enforcing the boundary condi -
tions of displacement and stress continuity at the defect, thereby 
Fig. 2. The architecture of PTUNet.
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
4 

## PDF页 5

obtaining the solutions for the scattered field. For the displacement 
boundary condition, we have: 
u
inc
+ u
>
= u
<
, 0 < z < 2 b . (16) 
And for stress boundary condition: 
 
σ
inc
+ σ
>
)
⋅ ̂e
r
= σ
<
⋅ ̂e
r
, 0 < z < 2 b , (17) 
where ̂e
r 
is the unit vector in the radial direction. Substituting the 
displacement and stress components into the boundary conditions and 
solving for the coefficients yield the solutions for the scattered field.
2.1.2. Kirchhoff – Love plate theory
Through 3D theory, the scattered displacement fields in r , θ , and z 
directions can be obtained, corresponding to the scattered S0 wave, 
scattered SH wave, and scattered A0 wave, respectively. In this paper, 
we focus on the scattering of the S0 wave in z direction (out-of-plane 
displacement) after incidence. The out-of-plane displacement informa -
tion is mainly provided by the scattered A0 mode. Therefore, the 
Kirchhoff – Love plate theory serves as an alternative approach for 
investigating the scattering problem. This model primarily captures the 
bending behavior of plates under small transverse loads, providing a 
foundation for analyzing the propagation of flexural waves within the 
structure [ 38 ]. The governing equation of the Kirchhoff – Love plate 
theory for an isotropic, homogeneous thin plate under free vibrations 
can be expressed as: 
D ∇
4
u ( x , y , t ) + ρ h
∂
2
u ( x , y , t )
∂ t
2
= 0 , (18) 
where 
D =
E ( 2 h )
3
12 ( 1   ν
2
)
(19) 
is the flexural stiffness of the plate, 
∇
4
=
∂
4
∂ x
4
+ 2
∂
4
∂ x
2
y
2
+
∂
4
∂ y
4
(20) 
is the bi-harmonic operator and u ( x , y , t ) represents the out-of-plane 
displacement at coordinates ( x , y ) at time t . After transformation, with 
ζ =
̅̅̅ ̅
ρ h
D
4
√
as a variable, Eq. (18) can be expressed as: 
∇
4
u ( x , y , t ) + ζ
∂
2
u ( x , y , t )
∂ t
2
= 0 . (21) 
Thus, the scattered A0 wave displacement can be obtained by solving 
Eq. (21) .
2.2. Principle of physics-enhanced TransUNet (PTUNet)
The proposed TransUNet architecture, as schematically illustrated in 
Fig. 2(a) , synergistically integrates the complementary strengths of UNet 
and Transformer models to address the multi-scale challenges in guided 
wave scattered wavefield reconstruction. UNet, as a classical encoder- 
decoder architecture, excels in image segmentation and feature extrac -
tion, making it well-suited for wavefield reconstruction [ 47 ]. Mean -
while, Transformer model, with its self-attention mechanism, effectively 
captures global dependencies, providing advantages in enhancing the 
representation capability of guided wave scattered fields [ 48 ]. This 
hybrid design synergistically integrates UNet ’ s local pattern recognition 
capacity for wave scattering identification with Transformer ’ s global 
dependency modeling for long-range wave propagation simulation.
TransUNet can obtain the scattered wavefield generated by defects 
with different geometric shapes and depth configurations. As depicted in 
Fig. 2(a) , the network employs a symmetric encoder-decoder structure 
with four hierarchical processing stages. The encoder pathway performs 
progressive downsampling through sequential operations: dual 3 × 3 
convolutional layers (LeakyReLU-activated), a 2 × 2 max-pooling layer, 
and a Transformer block incorporating multi-head attention mecha -
nisms. These attention modules systematically capture long-range inter- 
layer dependencies across spatiotemporal coordinates, a critical 
enhancement over conventional convolutional approaches for wavefield 
prediction. The decoder executes mirrored upsampling via 2 × 2 
transposed convolutions followed by feature concatenation and dual 3 
× 3 convolutions, with skip connections bridging encoder-decoder 
stages to preserve high-resolution scattering features. The detailed 
description of the Transformer block and the entire PETUNet architec -
ture can be found in Table 1 in Appendix . The architecture processes 
101 × 101 thickness maps as inputs, producing 101 × 101 × 101 
wavefield snapshots (400 mm × 400 mm spatial domain, dx = dy =
4 mm ) spanning 75 μ s to 175 μ s with 1 μ s temporal resolution. The 
network ’ s final output layer is configured with 101 channels, each 
representing one temporal snapshot. In this way, time is modeled as an 
additional “ channel dimension, ” and the network jointly learns the 
spatiotemporal evolution in a single forward pass. The fundamentals of 
neural network architectures, including layer-wise forward propagation 
mechanisms, are well-documented in literatures and therefore omitted 
for brevity. For the thickness input configuration indexed by i , let 
u
i
( x , y , t ) denote the reference wavefield obtained through FE simula -
tions, while u
*
i
( x , y , t ) represents the network prediction. The mean 
squared error (MSE) loss function governing the optimization process is 
formulated as: 
L
D
=
1
N
D
∑
N
D
i = 1
 
u
i
( x , y , t )   u
∗
i
( x , y , t )
)
2
, (22) 
where N
D 
corresponds to the spatiotemporal sampling points across the 
101 × 101 × 101 domain. In this study, we compute L
D 
using all points 
within the domain. We refer to L
D 
as the data-driven loss since it solely 
represents the discrepancy between the predictions and the references. 
By providing labeled samples and iteratively minimizing L
D
, Tran -
sUNet learns the relationship between different defects and the corre -
sponding scattered wavefields.
As previously discussed, a purely data-driven network heavily de -
pends on the dataset and may produce predictions that violate physical 
laws. Therefore, unlike conventional PINNs which require no training 
data but are limited to a fixed input model, we enhance the original 
TransUNet by introducing Kirchhoff-Love equation constraints. This 
approach retains the core advantages of data-driven learning while 
incorporating physical constraints as corrections. The physics guidance 
ensures that the network not only fits the temporal outputs but also 
learns to preserve physical consistency across the entire time domain. 
Specifically, the network is initially trained using only data-driven 
constraints. Once the data-driven loss L
D 
ceases to decrease, the 
physical constraints are introduced, and the network undergoes a second 
round of training with both constraints applied simultaneously. In the 
early training stage, the data constraint is more effective in guiding the 
network to capture the main wavefield distribution, while the physics 
constraint provides long-term physical consistency. This strategy can 
obviously reduce some computational cost. The training process of 
PTUNet is illustrated in Fig. 2(b) . Similar to the data-driven loss, the 
physics-driven loss L
P 
is expressed as: 
L
P
=
1
N
P
∑
N
P
i = 1
(
∇
4
u
∗
i
( x , y , t ) + ζ
∂
2
u
∗
i
( x , y , t )
∂ t
2
) 2
, (23) 
where N
P 
is the spatiotemporal points randomly sampled from the 
computational domain. Partial derivative terms in the governing equa -
tion are obtained via the central FD scheme. The calculation process of 
the partial derivatives at u
*
i
( x , y , t ) is described in Appendix . During the 
physics-constrained retraining phase, the composite loss function 
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
5 

## PDF页 6

integrates both data-driven and physics- driven components as: 
L = L
D
+ λ L
P
, (24) 
where λ denotes the adaptive weighting hyperparameter that balances 
the relative contributions of the data-driven and physics-driven con -
straints, ensuring numerical comparability between these two loss 
components. The selection of λ should be adjusted flexibly based on 
specific results, and the optimal value is determined through a process of 
trial and error.
3. Simulations and results
In this section, simulations were conducted on aluminum plates with 
irregularly shaped defects of varying depths to validate the enhance -
ment effect of the physical constraints in the proposed method and to 
assess its effectiveness in solving the scattered wavefields.
3.1. Numerical simulation setup
We simulated the scattering of S0 Lamb waves by irregular defects in 
an aluminum plate using FE software. As shown in Fig. 3 , the aluminum 
plate measures 800 mm in length, 800 mm in width, and 1.5 mm in 
thickness, with a density of ρ = 2700 kg/m
3
, Young ’ s modulus E = 70 . 7 
GPa, and Poisson ’ s ratio ν = 0 . 33. The defects, characterized by arbi -
trary irregular shapes and varying depths, are centered at (200, 400) and 
constrained within a 70 mm × 70 mm square. For each specific defect, 
the thickness remains constant. A symmetric dual-sensor excitation 
method was employed to generate S0 Lamb wave. A piezoelectric 
transducer with a center frequency of 100 kHz was positioned at (600, 
400), while another identical transducer was attached to the opposite 
side of the plate. To mitigate boundary reflections, 200 mm-wide 
perfectly matched layers (PML) were applied around the plate. In FE 
simulation, the excitation signal was a five-cycle Hanning-windowed 
sinusoidal function with a center frequency of 100 kHz. The time step 
was set to 1 × 10
  7 
s, with a total simulation duration of 2 . 5 × 10
  4 
s. In 
this study, a maximum mesh size of 5 mm was adopted, with local 
refinement applied near the defect. This resolution, together with local 
refinement, allowed both modes to be reasonably resolved while keep -
ing the computational cost manageable.
Fig. 4 presents the FE simulation results of a blind hole and their 
comparisons with 3D theory. The blind hole is located at (200, 400), 
with a radius of 25 mm, and a remaining thickness of 0.75 mm. The 
simulation provides 3D scattered wavefield data. Fig. 4(a) shows an out- 
of-plane displacement snapshot from FE simulation at 150 μ s . As 
observed in Fig. 4(a) , the incident S0 wave has propagated to the left 
boundary of the plate, generating scattered waves (S0-A0 waves) upon 
interacting with the defect. Fig. 4(b) demonstrates that the far-field 
amplitudes of out-of-plane displacement obtained from FE simulations 
aligns well with the theoretical predictions of 3D theory. This agreement 
validates the accuracy of FE parameter settings and confirms the reli -
ability of our simulations in capturing the scattering behavior of S0 
Lamb waves.
We extracted the out-of-plane displacement within a square region 
spanning x = 0 to 400 mm and y = 200 to 600 mm over a time window 
of 75 μ s to 175 μ s . Since the S0 Lamb wave reaches the defect region at 
approximately 75 μ s , the preceding wavefield does not contain any 
scattering information. To better focus on the relevant data for solving 
the scattering problem, we apply spatial and temporal truncation, 
ensuring that the extracted wavefield primarily captures the useful 
scattering characteristics. The resulting wavefield dataset had di -
mensions of 101 × 101 × 101 in x , y , and t . A total of 400 random defect 
Fig. 3. Simulation Settings.
Fig. 4. The scattering results of blind holes. (a) A snapshot of FE simulation. (b) Comparison of far-field amplitudes of scattered waves (black arrow represents 
incident direction).
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
6 

## PDF页 7

scattering simulations were performed to generate the dataset for 
network training, with 350 samples used for training and the remaining 
50 samples for testing. The remaining thickness of the defect ranges 
from 0.3 mm to 1 mm. Fig. 5 presents three different defects from the 
testing set, with increasing remaining thicknesses of 0.3655 mm, 0.5000 
mm, and 0.7322 mm, respectively. In the following sections, we use 
these three defect cases to evaluate the capability of the proposed 
method in solving the scattered wavefields.
3.2. Verification of the enhancement effect of physical constraints
To evaluate the enhancement effect of physical constraints on solv -
ing the scattered wavefield, we compared the training process and test 
results of TransUNet, which relies solely on data constraints, with 
PTUNet, which incorporates both data and physical constraints. Fig. 6
illustrates the training loss curves for both methods. The learning rate is 
0.0001, and the batch size is set to 1. During the first 300 iterations, 
training is conducted using only data constraints. As the data-driven loss 
begins to plateau at the 300th iteration, physical constraint is introduced 
with 2000 random collocation points for each time snapshot ( N
P
=
2000, 20 % of the total points) and the weight of 0.01 ( λ = 0 . 01). We 
determine the optimal weight values through grid searches over λ = 0 . 1, 
0 . 01, and 0 . 001. As shown in Fig. 6(a) , the embedding of physical 
constraints effectively accelerates the training process and reduces the 
overall training loss. Fig. 6(b) demonstrates that the physics-driven loss 
also decreases with iterations and eventually stabilizes after embedding 
the physical constraint.
To further evaluate the generalization capability of PTUNet, we 
compared the relative ℓ
2 
error of the two methods on the testing set. The 
relative ℓ
2 
error is computed as follows: 
Fig. 5. Thickness distribution maps of three defects in the testing set.
Fig. 6. Training loss curves. (a)Data-driven loss. (b) Physics-driven loss.
Fig. 7. The comparison of relative error on testing set.
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
7 

## PDF页 8

ℓ
2
=
⃦
⃦
u
i
( x , y , t )   u
*
i
( x , y , t )
⃦
⃦
2
‖ u
i
( x , y , t )‖
2
. (25) 
For each sample in the testing set, there are 101 time-domain 
snapshots of the scattered wavefield. We compute the ℓ
2 
error for 
each sample by averaging its relative ℓ
2 
errors across all time steps, as 
shown in Fig. 7 . The results indicate that PTUNet consistently achieves 
lower overall error compared to TransUNet, demonstrating superior 
performance on the testing data and enhanced generalization ability. 
This further validates the effectiveness of incorporating physical con -
straints, which not only guide the model to adhere to physical laws and 
accelerate convergence but also reduce dependency on training data. 
Consequently, for solving wave propagation problems and other 
physics-informed tasks, deep learning models with integrated physical 
constraints can effectively improve both accuracy and generalization.
3.3. Simulation results of PTUNet
3.3.1. Comparisons of wavefield
After validating the effectiveness of incorporating physical con -
straints, this section focuses on solving the S0 Lamb wave scattering 
process using PTUNet. Figs. 8 to 10 correspond to the scattered wave -
fields of the three randomly generated defects shown in Fig. 5(a), (b), 
and (c) , respectively. In Fig. 8 , (a1) to (a4) present the out-of-plane 
displacements obtained from 3D FE simulations, serving as references 
for the scattered wavefields. Fig. 8 (b1) to (b4) and (c1) to (c4) illustrate 
the PTUNet predictions and their respective error distributions relative 
to FE solutions. To comprehensively capture the wave propagation and 
scattering dynamics, we examine the wavefields at four distinct time 
instants — 95 μ s , 115 μ s , 135 μ s , and 155 μ s . As shown in Fig. 8 (a1), at 95 
μ s , S0 Lamb wave reaches the defect, initiating the formation of scat -
tered waves (S0-A0). By 115 μ s , S0 wave continues propagating, while 
the scattered waves radiate outward from the defect. At this stage, 
another wave emerges on the right side of the computational domain, 
corresponding to the A0 mode generated during excitation. At 135 μ s , as 
depicted in Fig. 8 (a3), S0 Lamb wave is on the verge of exiting the 
domain, while the scattered waves have begun interacting with A0 
mode, resulting in wave interference. By 155 μ s , the originally excited S0 
Lamb wave has completely exited the domain, leaving behind only the 
scattered waves and the approaching A0 mode, which will interact with 
the defect and generate secondary scattered waves (A0-A0), further 
increasing the complexity of the wavefield. It is noteworthy that as wave 
propagation progresses, the amplitude of the excited S0 wave gradually 
attenuates. In contrast, the amplitude of the scattered waves initially 
increases as they form and then gradually diminishes over time. Addi -
tionally, the wavefields exhibit anisotropic amplitude variations across 
different propagation angles, highlighting the influence of defect 
asymmetry on the scattering characteristics. This intrinsic correlation 
between defect geometry and the scattered wavefield underscores the 
feasibility of inferring defect properties through wave scattering 
analysis.
The comparisons between FE simulations and predictions indicate 
that PTUNet effectively captures the scattered wavefield throughout 
wave propagation. The wave propagation obtained from PTUNet closely 
align with those from FE simulations, demonstrating that the proposed 
method has successfully learned the scattering characteristics. As 
observed in the error distributions from Figs. 8 to 10 , discrepancies are 
primarily concentrated around the defect regions and theirs surrounding 
Fig. 8. The comparisons of the scattered wavefields of the defect shown in Fig. 5(a) .
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
8 

## PDF页 9

wave interference zones. Additionally, as wave propagation, the errors 
tend to increase in the far-field region. This may be attributed to PTU -
Net ’ s reduced prediction accuracy at longer time steps during training or 
the inherently complex nature of wave scattering induced by the defect. 
PTUNet needs to capture intricate physical details, and its predictive 
capability faces challenges in fully reconstructing all fine-scale features. 
Despite these local errors, the results presented in Figs. 8 to 10 confirm 
that PTUNet can solve the scattered wavefields of defects with different 
shapes and thicknesses. The model exhibits reasonable generalization 
ability and maintains high physical consistency, making it a promising 
approach for wave scattering analysis in complex structures.
3.3.2. Comparisons of signals and far-field amplitudes of the scattered 
waves
After obtaining the scattered wavefield for irregular defects, the 
time-domain signal at any arbitrary location within the solution domain 
can be extracted. To systematically analyze the scattering characteris -
tics, we define the coordinate (200, 400), the center of the domain, as 
the origin and uniformly distribute 360 receiver points along a circular 
trajectory with a radius of 160 mm. This configuration enables the 
acquisition of time-domain signals at each receiver location, facilitating 
a detailed examination of the wave propagation and scattering behavior. 
Fig. 11 presents a comparative analysis of the scattered signals corre -
sponding to the three defects shown in Fig. 5(a), (b), and (c) . The signals 
are sampled at 12 angular positions with 30
◦
intervals, providing a 
representative overview of the wave propagation characteristics. As the 
observation angle increases, the propagation distance of S0 mode 
initially increases and subsequently decreases symmetrically about 
180
◦
, resulting in an arc-like distribution of the signals centered at 180
◦
. 
When S0 wave reaches the defect, scattered A0 waves (S0-A0) are 
generated and propagate outward around the defect. Since the propa -
gation paths to the receiver locations are nearly identical, the scattered 
A0 waves are expected to arrive at approximately the same time across 
different angular positions. The results in Fig. 11 further corroborate this 
observation, further validating the physical consistency of the scattering 
process.
From a global perspective, the signals predicted by PTUNet exhibit a 
good agreement with FE simulations. In the S0 wave region where 
scattering has not yet occurred, PTUNet ’ s predictions align almost 
perfectly with FE, demonstrating the model ’ s ability to accurately cap -
ture the undisturbed wave propagation. After approximately 135 μ s , 
when the scattered A0 wave reaches the receiver locations, slight dis -
crepancies in amplitude emerge between predictions and FE results. 
Nevertheless, the overall wave propagation trend remains consistent 
across both methods. These minor deviations may be attributed to 
PTUNet ’ s inherent limitations in learning fine-scale scattering features, 
which could lead to reduced accuracy in capturing wave interactions.
To further examine the scattering characteristics, we extracted the 
peak amplitude of the scattered A0 waves at different angular positions, 
yielding the results of the far-field amplitudes of the scattered waves 
presented in Fig. 12 . The results illustrate that the predicted scattering 
energy distribution closely aligns with FE simulations, exhibiting a 
similar scattering pattern. This further substantiates PTUNet ’ s ability to 
generalize across different scatterers. Notably, in regions where the 
scattering features exhibit pronounced variations, such as near 90
◦
and 
270
◦
in Fig. 12(b) , PTUNet successfully captures these variations with 
good fidelity. While some discrepancies are observed in specific angular 
ranges, such as between 150
◦
and 180
◦
in Fig. 12(a) and (c) , the overall 
Fig. 9. The comparisons of the scattered wavefields of the defect shown in Fig. 5(b) .
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
9 

## PDF页 10

trend remains well-aligned with FE simulations. These deviations may 
stem from several factors such as the high complexity of certain scat -
tering behaviors. Overall, PTUNet demonstrates a considerable capacity 
for predicting the propagation characteristics of scattered waves. Future 
efforts could focus on refining the physical constraint mechanisms to 
further enhance the predictive performance and generalization of the 
model.
Fig. 10. The comparisons of the scattered wavefields of the defect shown in Fig. 5(c) .
Fig. 11. The comparisons of the signals of the three defects shown in Fig. 5 .
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
10 

## PDF页 11

4. Experiments and results
4.1. Experiment setup
To experimentally validate the performance of the proposed method 
in solving the scattered wavefield, we conducted a field scanning 
experiment using a Polytec SLDV. The experimental setup is shown in 
Fig. 13 . A 6061 aluminum plate with dimensions of 800 × 800 × 1.5 mm 
was securely mounted on an optical platform. To minimize boundary 
reflections, damping clay was applied along the plate ’ s edges. Specif -
ically, the clay was shaped into a wedge, gradually tapering in thickness 
from the plate ’ s edge toward its center, as illustrated in Fig. 13(b) . This 
design effectively reduces secondary reflections caused by wave in -
teractions with the damping material. An irregularly shaped defect with 
a residual thickness of 0.5 mm was precisely machined onto the 
aluminum plate using a milling process, as shown in Fig. 13(c) . The 
spatial coordinates of the defect were kept consistent with those used in 
FE simulations, ensuring a rigorous evaluation of the predictive accu -
racy of the proposed method. Two identical 100 kHz piezoelectric 
transducers were symmetrically mounted on the aluminum plate at a 
distance of 400 mm from the defect center to generate a pure S0 Lamb 
wave. A five-cycle sinusoidal excitation signal with a frequency of 100 
kHz and an amplitude of 4 V peak to peak was generated by an arbitrary 
waveform generator. The signal was then amplified by a factor of 20 
using an Aigtek ATA-2022B dual-channel power amplifier before being 
applied to the two transducers. The SLDV probe was carefully adjusted 
to ensure that the emitted laser beam was as perpendicular as possible to 
the plate surface. To enhance the probe ’ s ability to capture vibrations, a 
reflective material was applied to the defect surface. The scanning re -
gion was defined as a 100 mm × 100 mm square centered on the defect. 
The SLDV operated at a sampling frequency of 6.25 MHz, with a total of 
5000 acquisition points. A band-pass filter was applied in the range of 
80 kHz to 120 kHz to isolate the relevant frequency components.
4.2. Comparisons of scattering results
We input the thickness distribution of the defect into the pre-trained 
PTUNet model to generate the predicted scattered wavefields and 
compare them with FE results. As shown in Fig. 14 , PTUNet accurately 
captures the propagation of S0 waves and the scattering behaviors after 
interacting with the defect, showing good consistency with FE 
simulations.
Fig. 12. The comparisons of the far-field amplitudes of the scattered waves of three defects shown in Fig. 5 (black arrow represents incident direction).
Fig. 13. Setup of the SLDV acoustic field scanning experiment.
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
11 

## PDF页 12

Through SLDV scanning, we obtained the scattered wavefields of the 
defect in experiment. Assuming 360 receivers are uniformly distributed 
along a circular path with a radius of 100 mm centered on the defect, the 
time-domain signal and its continuous wavelet transform (CWT) at the 
180
◦
receiving point are presented in Fig. 15 . Given the wave structure 
of S0 mode, its out-of-plane displacement is relatively small, making it 
challenging to capture with SLDV. Consequently, as shown in Fig. 15(a) , 
the amplitude of S0 mode is significantly small. Based on the propaga -
tion distance, S0 wave is expected to appear around 90 μ s , while the 
scattered wave generated by the defect emerges near 135 μ s . The scat -
tered wave is more pronounced because, compared to S0 mode, A0 
mode exhibits a larger out-of-plane displacement. The most dominant 
wave component, A0 mode generated by the excitation, appears around 
240 μ s with the highest amplitude. The CWT of the signal further reveals 
Fig. 14. The comparisons of the scattered wavefields of the defect in experiments.
Fig. 15. Analysis of the experimental signal. (a) Time domain signal at the receiving point of 180
◦
. (b) Continuous wavelet transform analysis.
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
12 

## PDF页 13

its time – frequency characteristics, as shown in Fig. 15(b) . The frequency 
is mainly distributed around 100 kHz. Due to the dominant amplitude of 
the excitation-induced A0 mode, the maximum energy is observed 
around 300 μ s , aligning well with theoretical expectations.
Fig. 16(a) presents the time-domain signals obtained from 12 re -
ceivers, which uniformly distributed at 30
◦
intervals. These signals are 
utilized to derive the far-field amplitudes of the scattered waves ob -
tained from SLDV scanning. The comparisons of FE simulation, PTU -
Net ’ s prediction, and SLDV scanning are shown in Fig. 16(b) . Despite the 
relatively smooth and simple geometry of the defect boundary, the 
scattered wavefield exhibits significant complexity, as evidenced by 
obvious amplitude variations in the scattering result. The PTUNet pre -
dictions demonstrate good agreement with FE simulations, effectively 
and comprehensively capturing the intricate scattering characteristics. 
The proposed method successfully reconstructs the majority of the 
scattering information, underscoring its capability to model wave in -
teractions with irregular defects accurately. Although discrepancies 
exist between the experimental scanning and FE simulations, the overall 
Fig. 16. (a) Experimental time-domain signals. (b) Comparison of far-field amplitude of scattered waves (black arrow represents incident direction).
Table 1 
Network architecture & hyperparameters.
Module Configuration Output size
Input Thickness map (1 channel) 1 × 101 ×
101
Encoder-1 Conv2d (1 → 16), LeakyReLU(0.01) 
Conv2d (16 → 16), LeakyReLU(0.01)
16 × 101 ×
101
Transformer- 
1
MultiheadAttention (embed = 16, heads = 4), FFN 
(16 → 32 → 16), LayerNorm
16 × 101 ×
101
Encoder-2 Conv2d (16 → 32), LeakyReLU(0.01) 
Conv2d (32 → 32), LeakyReLU(0.01)
32 × 50 ×
50
Transformer- 
2
MultiheadAttention (embed = 32, heads = 4), FFN 
(32 → 32 → 32), LayerNorm
32 × 50 ×
50
Encoder-3 Conv2d (32 → 64), LeakyReLU(0.01) 
Conv2d (64 → 64), LeakyReLU(0.01)
64 × 25 ×
25
Transformer- 
3
MultiheadAttention (embed = 64, heads = 4), FFN 
(64 → 32 → 64), LayerNorm
64 × 25 ×
25
Encoder-4 Conv2d (64 → 128), LeakyReLU(0.01) 
Conv2d (128 → 128), LeakyReLU(0.01)
128 × 13 ×
13
Transformer- 
4
MultiheadAttention (embed = 128, heads = 4), 
FFN (128 → 32 → 128), LayerNorm
128 × 13 ×
13
Bottleneck Conv2d (128 → 256), LeakyReLU(0.01) 
Conv2d (256 → 256), LeakyReLU(0.01)
256 × 7 × 7
Decoder-1 ConvTranspose2d (256 → 128) + skip 
Conv2d (256 → 128), LeakyReLU(0.01) 
Conv2d (128 → 128), LeakyReLU(0.01)
128 × 14 ×
14
Decoder-2 ConvTranspose2d (128 → 64) + skip 
Conv2d (128 → 64), LeakyReLU(0.01) 
Conv2d (64 → 64), LeakyReLU(0.01)
64 × 28 ×
28
Decoder-3 ConvTranspose2d (64 → 32) + skip 
Conv2d (64 → 32), LeakyReLU(0.01) 
Conv2d (32 → 32), LeakyReLU(0.01)
32 × 56 ×
56
Decoder-4 ConvTranspose2d (32 → 16) + skip 
Conv2d (32 → 16), LeakyReLU(0.01) 
Conv2d (16 → 16), LeakyReLU(0.01)
16 × 112 ×
112
Output Conv2d (16 → 101), bilinear resize to 101 × 101 101 × 101 
× 101
Fig. 17. Comparison of governing equation residuals before (black) and after 
(red) training.
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
13 

## PDF页 14

trend remains consistent. Considering factors such as the inherently 
weak out-of-plane scattering amplitude, unavoidable boundary re -
flections, machining tolerances of the defect, and various experimental 
noise sources, it is reasonable to conclude that the SLDV experimental 
results are in qualitative agreement with both FE simulations and 
PTUNet predictions, reinforcing their mutual validation.
5. Discussion
5.1. Advantages of the proposed method
By integrating the local feature extraction capabilities of UNet with 
the global modeling power of Transformer, and further constraining the 
solution with Kirchhoff-Love plate theory, PTUNet demonstrates 
outstanding performance in solving S0 Lamb wave scattering induced by 
irregular defects in plate structures. Notably, the irregular defect size is 
approximately one-tenth of the wavelength of the excitation signal, 
leading to highly intricate scattering characteristics, including mode 
conversion and other fine wavefield details. Through the enhancement 
of physical constraints, PTUNet achieves accurate predictions of subtle 
scattering features with a limited training dataset, yielding a lower mean 
relative ℓ
2 
error on the testing set compared to purely data-driven 
training approach. Unlike BEM or hybrid numerical methods, PTUNet 
eliminates the need for repeated modeling of different configurations 
and does not require mesh discretization, thereby avoiding the sub -
stantial computational costs associated with high-frequency simula -
tions. In computing the partial derivatives required for physical 
constraints, we adopt FD methods instead of automatic differentiation. 
This choice is driven by the fact that employing automatic differentia -
tion would necessitate including spatial coordinates x, y and temporal 
coordinate t as additional network inputs. Consequently, the input 
dimensionality would increase from 2D thickness representations to a 
4D dataset incorporating spatial and temporal information, significantly 
escalating computational complexity.
It is crucial to highlight that the proposed method offers valuable 
insights that can inspire solutions for a broader range of problems. A 
particularly promising application lies in defect inversion based on 
scattering information. By leveraging the enhancement of physical 
constraints, it becomes possible to reconstruct the thickness and shape of 
defects approximately one-tenth of the incident wavelength, represent -
ing an exciting research avenue. Furthermore, the inherent generaliz -
ability and scalability of this approach pave the way for its extension to 
other fields, as the underlying physical model can always be associated 
with appropriate governing equations. This versatility underscores the 
method’s potential as a powerful tool for advancing interdisciplinary 
research and fostering new developments in scientific and engineering 
applications.
5.2. Limitations and future works
In this study, the defect models remain relatively simplified, as the 
residual thickness within each defect is assumed to be uniform. This 
assumption differs significantly from real-world defects, which always 
exhibit complex and irregular thickness variations. The primary reason 
for this simplification is the limited thickness (1.5 mm) of the tested 
plate, which inherently constrains the range of residual thickness vari -
ations. Minor variations in thickness are unlikely to induce substantial 
changes in scattering characteristics. Furthermore, due to the consid -
erable computational cost of 3D FE simulations, it is time-consuming to 
generate a much larger dataset for training. Capturing the subtle vari -
ations in the scattered wavefield caused by small thickness changes re -
mains a formidable challenge. To address these limitations, we propose 
two key research directions. First, we plan to explore the potential of 
neural operators for solving scattered wavefields, aiming to enhance 
feature extraction and self-learning capabilities while reducing the 
network complexity. Second, we plan to incorporate additional and 
more precise governing equations to better describe wave propagation, 
thereby improving the model’s ability to capture intricate scattering 
behaviors with higher fidelity.
Furthermore, our experimental results are not yet perfect. This is 
primarily due to challenges in exciting a pure S0 mode during the 
experiment. Several factors may contribute to the discrepancies in the 
results, including the asymmetric arrangement of the excitation trans -
ducers, differences in material properties of the aluminum plate 
compared to the simulations, and unavoidable boundary reflections. 
Additionally, the out-of-plane displacement of S0 mode is relatively 
small compared to A0 mode, which means that the scattered wave in -
formation may be submerged in noise, making it difficult to capture with 
high precision.
Finally, this study has primarily focused on solving the out-of-plane 
displacement of the scattered wavefield. Moving forward, we intend to 
explore more general wave equations to address the solution of 3D 
scattering characteristics. Additionally, we will investigate alternative 
implementations of physical constraints, such as using integral form 
constraints or leveraging Chebyshev spectral methods to accelerate the 
calculation of higher-order derivatives. These advancements will 
enhance the accuracy and generalization of the model in solving com -
plex scattering problems.
6. Summary
In this study, we propose a physics-enhanced TransUNet (PTUNet) 
framework to solve the scattered wavefield of irregular defects in plate 
structures. This method possesses the capability of extracting local fea -
tures and modeling global characteristics of the scattered wavefield. 
Moreover, with the physical constraints imposed by Kirchhoff-Love 
plate theory, it enables accurate capture of fine-scale features in com -
plex scattered wavefields. We conducted simulations on three sets of 
randomly generated defects and compared the results with FE simula -
tions. The proposed method effectively predicts the propagation of S0 
Lamb waves and their mode conversion upon interacting with defects, 
capturing the intricate scattering characteristics. The predicted time- 
domain signals and the far-field amplitudes of scattered waves from 
PTUNet align well with FE results. Additionally, we performed acoustic 
field scanning experiments using SLDV to measure the out-of-plane 
displacement of scattered waves from a random defect on a 1.5 mm- 
thick aluminum plate. The scattering result predicted by PTUNet ex -
hibits a reasonable agreement with both FE simulation and experimental 
measurement. These simulation and experimental results validate the 
effectiveness of the proposed method in solving S0 Lamb wave scat -
tering problems. Future work includes exploring neural operators to 
improve predictive performance, investigating alternative governing 
equations as physical constraints to achieve more accurate 3D wavefield 
solutions, and studying different implementations of physical con -
straints for further improvements.
CRediT authorship contribution statement
Linfeng Wang: Writing – original draft, Visualization, Methodology. 
Hongyan Zhang: Visualization, Software, Methodology. Zhen Zhang: 
Validation, Methodology. He Sun: Software, Formal analysis. Xiao 
Ying: Validation, Software. Jing Hou: Investigation, Funding acquisi -
tion. Jian Li: Writing – review & editing, Supervision, Methodology. 
Zhoumo Zeng: Writing – review & editing, Supervision, Funding 
acquisition. Yang Liu: Writing – review & editing, Supervision, Funding 
acquisition.
Declaration of competing interest
The authors declare that they have no known competing financial 
interests or personal relationships that could have appeared to influence 
the work reported in this paper.
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
14 

## PDF页 15

Acknowledgments
This work was supported in part by the National Science Foundation 
of China under Contract No. 12474459, in part by the Natural Science 
Foundation of Zhejiang Province under Contract No. LY23E010001.
Appendix A 
The detailed description of PETUNet architecture is presented in Table 1 .
The partial derivatives at u
*
i
( x , y , t ) using the central finite difference scheme are described as 
∂
4
u
*
i
∂ x
4
=
u
*
i
( x + 2 , y , t )   4 u
*
i
( x + 1 , y , t ) + 6 u
*
i
( x , y , t )   4 u
*
i
( x   1 , y , t ) + u
*
i
( x   2 , y , t )
( Δ x )
4
, (A.1) 
∂
4
u
*
i
∂ y
4
=
u
*
i
( x , y + 2 , t )   4 u
*
i
( x , y + 1 , t ) + 6 u
*
i
( x , y , t )   4 u
*
i
( x , y   1 , t ) + u
*
i
( x , y   2 , t )
( Δ y )
4
, (A.2) 
∂
4
u
*
i
∂ x
2
∂ y
2
=
1
( Δ x )
2
( Δ y )
2
[ u
*
i
( x + 1 , y + 1 , t ) + u
*
i
( x + 1 , y   1 , t ) + u
*
i
( x   1 , y + 1 , t )
+ u
*
i
( x   1 , y   1 , t )   2 u
*
i
( x   1 , y , t )   2 u
*
i
( x , y + 1 , t )   2 u
*
i
( x , y   1 , t )
+ 4 u
*
i
( x , y , t )] ,
(A.3) 
∂
2
u
*
i
∂ t
2
=
u
*
i
( x , y , t + 1 )   2 u
*
i
( x , y , t ) + u
*
i
( x , y , t   1 )
( Δ t )
2
. (A.4) 
To validate the effectiveness of FD approximations in the calculation of physics-driven loss, we compare the pre-training and post-training gov -
erning equation residuals at 2000 randomly sampled coordinates for a representative time snapshot, as shown in Fig. 17 . The residuals converge to 
near-zero values upon training completion, demonstrating a reduction of over two orders of magnitude compared to its pre-training level.
Data availability
The data that support the findings of this study are available from the 
corresponding author upon reasonable request.
References
[1] R.L. Joesph, Ultrasonic Waves in Solid Media, Cambridge University Press, 2003 .
[2] C.A. Chua, P. Cawley, P.B. Nagy, Scattering of the fundamental shear guided wave 
from a surface-breaking crack in plate-like structures, IEEE Trans. Ultrason. 
Ferroelectr. Freq. Control 66 (2019) 1887 – 1897, https://doi.org/10.1109/ 
TUFFC.2019.2932227 .
[3] P. Huthwaite, Guided wave tomography with an improved scattering model, Proc. 
Royal. Soc. A 472 (2016) 20160643, https://doi.org/10.1098/rspa.2016.0643 .
[4] A. Spada, M. Capriotti, F. Lanza di Scalea, Global – local model for three- 
dimensional guided wave scattering with application to rail flaw detection, Struct. 
Health Monit. 21 (2) (2022) 370 – 386, https://doi.org/10.1177/ 
14759217211000863 .
[5] J.D. Achenbach, Wave Propagation in Elastic Solids, North-Holland, Amsterdam, 
1973 .
[6] B. Auld, Acoustic Fields and Waves in Solids, Krieger, Malabar, Florida, 1990 .
[7] O. Diligent, T. Grahn, A. Bostr ¨om, P. Cawley, M.J.S. Lowe, The low-frequency 
reflection and scattering of the S0 Lamb mode from a circular through-thickness 
hole in a plate: Finite element, analytical and experimental studies, J. Acoust. Soc. 
Am. 112 (6) (2002) 2589 – 2601, https://doi.org/10.1121/1.1512292 .
[8] T. Grahn, Lamb wave scattering from a circular partly through-thickness hole in a 
plate, Wave Motion 37 (1) (2003) 63 – 80, https://doi.org/10.1016/S0165-2125 
(02)00051-3 .
[9] L. Moreau, M. Caleap, A. Velichko, P.D. Wilcox, Scattering of guided waves by 
through-thickness cavities with irregular shapes, Wave Motion 48 (7) (2011) 
586 – 602, https://doi.org/10.1016/j.wavemoti.2011.04.010 .
[10] F.B. Cegla, A. Rohde, M. Veidt, Analytical prediction and experimental 
measurement for mode conversion and scattering of plate waves at non-symmetric 
circular blind holes in isotropic plates, Wave Motion 45 (3) (2008) 162 – 177, 
https://doi.org/10.1016/j.wavemoti.2007.05.005 .
[11] A. Sedaghati, F. Honarvar, M. Tabatabaeipour, A.N. Sinclair, Investigation of the 
scattering of Lamb waves from a generalized circular cavity by using Poisson/ 
Mindlin plate theories and numerical simulation, Proc. Inst. Mech. Eng. Part C J. 
Mech. Eng. Sci. 234 (2020) 152 – 170, https://doi.org/10.1177/ 
0954406219871073 .
[12] Z. Wang, Q. Leng, T. Li, Elastic wave scattering by flat-bottomed indentations on a 
plate, Int. J. Mech. Sci. 231 (2022) 107583, https://doi.org/10.1016/j. 
ijmecsci.2022.107583 .
[13] L.R. Rose, C.H. Wang, Mindlin plate theory for damage detection: Imaging of 
flexural inhomogeneities, J. Acoust. Soc. Am. 127 (2) (2010) 754 – 763, https://doi. 
org/10.1121/1.3277217 .
[14] B. Poddar, V. Giurgiutiu, Complex modes expansion with vector projection using 
power flow to simulate Lamb waves scattering from horizontal cracks and 
disbands, J. Acoust. Soc. Am. 140 (3) (2016) 2123 – 2133, https://doi.org/10.1121/ 
1.4963087 .
[15] B. Poddar, V. Giurgiutiu, Scattering of Lamb waves from a discontinuity: an 
improved: analytical approach, Wave Motion 65 (2016) 79 – 91, https://doi.org/ 
10.1016/j.wavemoti.2016.03.009 .
[16] M.F. Haider, B. Poddar, V. Giurgiutiu, Experimental validation of an analytical 
method to predict lamb wave scattering from a discontinuity, Smart Materi. Struct. 
28 (1) (2018) 015012, https://doi.org/10.1088/1361-665X/aae910 .
[17] M.F. Haider, R. Joseph, V. Giurgiutiu, B. Poddar, An efficient analytical 
global – local (AGL) analysis of the Lamb wave scattering problem for detecting a 
horizontal crack in a stiffened plate, Acta Mech. 231 (2020) 577 – 596, https://doi. 
org/10.1007/s00707-019-02555-z .
[18] M.F. Haider, M.Y. Bhuiyan, B. Poddar, B. Lin, V. Giurgiutiu, Analytical and 
experimental investigation of the interaction of Lamb waves in a stiffened 
aluminum plate with a horizontal crack at the root of the stiffener, J. Sound Vib. 
431 (2018) 212 – 225, https://doi.org/10.1016/j.jsv.2018.06.018 .
[19] Y. Shen, V. Giurgiutiu, WFR-2D: an analytical model for PWAS-generated 2D 
ultrasonic guided wave propagation, in: SPIE, 2014, https://doi.org/10.1117/ 
12.2044798 .
[20] Q. Deng, Z. Yang, Scattering of S0 Lamb mode in plate with multiple damage, Appl. 
Math. Model. 35 (1) (2011) 550 – 562, https://doi.org/10.1016/j. 
apm.2010.07.022 .
[21] L. Sun, J. Wei, C. Peng, W. Hao, F. Teng, L. Fan, L. Zhang, Q. Sui, M. Jiang, 
Ultrasonic guided wave-based probabilistic diagnostic imaging method with 
Single-Path-Scattering sparse reconstruction for Multi-damage detection in 
composite structures, Mech. Syst. Sig. Process. 223 (2025) 111858, https://doi. 
org/10.1016/j.ymssp.2024.111858 .
[22] C. Yeung, C.T. Ng, Time-domain spectral finite element method for analysis of 
torsional guided waves scattering and mode conversion by cracks in pipes, Mech. 
Syst. Sig. Process. 128 (2019) 305 – 317, https://doi.org/10.1016/j. 
ymssp.2019.04.013 .
[23] H. Gravenkamp, J. Prager, A.A. Saputra, C. Song, The simulation of Lamb waves in 
a cracked plate using the scaled boundary finite element method, J. Acoust. Soc. 
Am. 132 (3) (2012) 1358 – 1367, https://doi.org/10.1121/1.4740478 .
[24] J. Yue, P. Li, Numerical solution of the cavity scattering problem for flexural waves 
on thin plates: linear finite element methods, J. Comput. Phys. 497 (2024) 112606, 
https://doi.org/10.1016/j.jcp.2023.112606 .
[25] Y. Cho, J.L. Rose, An elastodynamic hybrid boundary element study for elastic 
guided wave interactions with a surface breaking defect, Int. J. Solids Struct. 37 
(30) (2000) 4103 – 4124, https://doi.org/10.1016/S0020-7683(99)00142-0 .
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
15 

## PDF页 16

[26] T. Maruyama, K. Kanda, S. Yamada, Three-dimensional hybrid SAFE-BEM for 
elastic guided-wave scattering in a plate with finite width, Eng. Anal. Bound. Elem. 
163 (2024) 426 – 435, https://doi.org/10.1016/j.enganabound.2024.03.027 .
[27] W. Shao, H. Sun, Q. Zhou, Y. Wang, X. Qing, A novel Lamb wave-based multi- 
damage dataset construction and quantification algorithm under the framework of 
multi-task deep learning, Struct. Health Monit. 23 (2) (2024) 1148 – 1169, https:// 
doi.org/10.1177/14759217231185051 .
[28] D. Ji, J. Lin, F. Gao, The high-sparsity Lamb wavefield reconstruction framework 
based on masked autoencoder, IEEE Trans. Instrum. Meas. 72 (2023) 1 – 10, https:// 
doi.org/10.1109/TIM.2023.3316259 .
[29] H. Zhang, J. Lin, J. Hua, T. Tong, Interpretable convolutional sparse coding method 
of Lamb waves for damage identification and localization, Struct. Health Monit. 21 
(4) (2022) 1790 – 1804, https://doi.org/10.1177/14759217211044806 .
[30] C. Humer, S. H ¨oll, C. Kralovec, M. Schagerl, Damage identification using wave 
damage interaction coefficients predicted by deep neural networks, Ultrasonics 
124 (2022) 106743, https://doi.org/10.1016/j.ultras.2022.106743 .
[31] S. Zhang, C.M. Li, W. Ye, Damage localization in plate-like structures using time- 
varying feature and one-dimensional convolutional neural network, Mech. Syst. 
Sig. Process. 147 (2021) 107107, https://doi.org/10.1016/j.ymssp.2020.107107 .
[32] M. Raissi, P. Perdikaris, G.E. Karniadakis, Physics-informed neural networks: a 
deep learning framework for solving forward and inverse problems involving 
nonlinear partial differential equations, J. Comput. Phys. 378 (2019) 686 – 707, 
https://doi.org/10.1016/j.jcp.2018.10.045 .
[33] L. Yang, X. Meng, G.E. Karniadakis, B-PINNs: Bayesian physics-informed neural 
networks for forward and inverse PDE problems with noisy data, J. Comput. Phys. 
425 (2021) 109913, https://doi.org/10.1016/j.jcp.2020.109913 .
[34] X. Meng, Z. Li, D. Zhang, G.E. Karniadakis, PPINN: Parareal physics-informed 
neural network for time-dependent PDEs, Comput. Methods Appl. Mech. Eng. 370 
(2020) 113250, https://doi.org/10.1016/j.cma.2020.113250 .
[35] L. Wang, H. Wang, L. Liang, J. Li, Z. Zeng, Y. Liu, Physics-informed neural 
networks for transcranial ultrasound wave propagation, Ultrasonics 132 (2023) 
107026 .
[36] H. Wang, J. Li, L. Wang, L. Liang, Z. Zeng, Y. Liu, On acoustic fields of complex 
scatters based on physics-informed neural networks, Ultrasonics 128 (2023) 
106872, https://doi.org/10.1016/j.ultras.2022.106872 .
[37] H. Sun, L. Peng, J. Lin, S. Wang, W. Zhao, S. Huang, Microcrack defect 
quantification using a focusing high-order SH guided wave EMAT: the physics- 
informed deep neural network GuwNet, IEEE Trans. Ind. Inf. 18 (5) (2021) 
3235 – 3247, https://doi.org/10.1109/TII.2021.3105537 .
[38] W. Zhou, Y.F. Xu, Damage identification for plate structures using physics- 
informed neural networks, Mech. Syst. Sig. Process. 209 (2024) 111111, https:// 
doi.org/10.1016/j.ymssp.2024.111111 .
[39] M.S. Go, H.K. Noh, J.H. Lim, Real-time full-field inference of displacement and 
stress from sparse local measurements using physics-informed neural networks, 
Mech. Syst. Sig. Process. 224 (2025) 112009, https://doi.org/10.1016/j. 
ymssp.2024.112009 .
[40] L. Lu, P. Jin, G. Pang, Z. Zhang, G.E. Karniadakis, Learning nonlinear operators via 
DeepONet based on the universal approximation theorem of operators, Nat. Mach. 
Intell. 3 (3) (2021) 218 – 229, https://doi.org/10.1038/s42256-021-00302-5 .
[41] J. He, S. Koric, D. Abueidda, A. Najafi, I. Jasiuk, Geom-deeponet: a point-cloud- 
based deep operator network for field predictions on 3d parameterized geometries, 
Comput. Methods Appl. Mech. Eng. 429 (2024) 117130, https://doi.org/10.1016/ 
j.cma.2024.117130 .
[42] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. 
Anandkumar, Fourier neural operator for parametric partial differential equations, 
arXiv preprint arXiv:2010.08895 (2020). https://doi.org/10.48550/arXiv.2010. 
08895 .
[43] C. Song, Y. Wang, High-frequency wavefield extrapolation using the Fourier neural 
operator, J. Geophys. Eng. 19 (2) (2022) 269 – 282, https://doi.org/10.1093/jge/ 
gxac016 .
[44] Z. Li, H. Zheng, N. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, 
A. Anandkumar, Physics-informed neural operator for learning partial differential 
equations, ACM/IMS J. Data Sci. 1 (3) (2024) 1 – 27, https://doi.org/10.1145/ 
3648506 .
[45] L. Wang, J. Li, S. Chen, Z. Fan, Z. Zeng, Y. Liu, Finite difference-embedded UNet for 
solving transcranial ultrasound frequency-domain wavefield, J. Acoust. Soc. Am. 
155 (3) (2024) 2257 – 2269, https://doi.org/10.1121/10.0025391 .
[46] J.D. Achenbach, Y. Xu, Wave motion in an isotropic elastic layer generated by a 
time-harmonic point load of arbitrary direction, J. Acoust. Soc. Am. 106 (1999) 
83 – 90, https://doi.org/10.1121/1.427037 .
[47] O. Ronneberger, P. Fischer, T. Brox, U-Net: convolutional networks for biomedical 
image segmentation, Med. Image Comput. Comput. Assist. Interv. (MICCAI) 
(2015), https://doi.org/10.1007/978-3-319-24574-4_28 .
[48] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Ł . Kaiser, 
I. Polosukhin, Attention is all you need, in: Advances in Neural Information 
Processing Systems, 2017, pp. 5998 – 6008 .
L. Wang et al.                                                                                                                                                                                                                                   Ultrasonics 159 (2026) 107852 
16 