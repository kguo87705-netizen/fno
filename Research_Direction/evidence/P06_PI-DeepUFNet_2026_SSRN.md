# P06_PI-DeepUFNet_2026_SSRN.pdf

## PDF页 1

Physics-Informed Deep U-Fourier Neural Network for Efficient 
Prediction of Guided Wavefield in Plates with Irregular Defects
Zhao Wang1, Linfeng Wang1, Xiao Ying1, Jin Hou2, and Yang Liu1,
1 State Key Laboratory of Precision Measuring Technology and Instruments, Tianjin 
University, Tianjin, 300072, China 
2 CNOOC Research Institute Company Limited, Beijing, 100028, China
 Corresponding author.
E-mail address: ultrasonicslab@tju.edu.cn (Y. Liu)
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 2

Abstract
Accurate prediction of the full-field ultrasonic guided wave evolution in plates 
with irregular defects is essential for quantitative nondestructive evaluation(QNDE). 
Traditional numerical solutions encounter efficiency bottlenecks while existing data-
driven models are mostly limited by simplified scalar acoustic equations which struggle 
to characterize dispersion effects in plates with variable thickness. This paper proposes 
a physics-informed deep U-Fourier neural network (PI-DeepUFNet) merging 
Kirchhoff-Love (KL) plate theory with neural operators for the efficient prediction of 
spatiotemporal wavefield in thin plates containing irregular defects. The framework 
combines the local multi-scale geometric feature extraction advantages of U-Net with 
the global spatiotemporal evolution modeling capability of 3D Fourier Neural 
Operators (FNO). By discretizing the fourth-order flexural wave equation into finite-
difference operator and explicitly embedding them into the loss function, joint 
optimization of data and physics is achieved. Numerical experiments show that the 
model accurately predicts the A0 mode defect scattering response under complex 
corrosion conditions. The average RMSE, RL2, and PCC for wavefield prediction on 
the test set reach , , and , respectively. Ablation 
experiments confirm that physical constraints effectively enhance the prediction 
stability and dynamical consistency of the model in complex scattering scenarios. 
Geometric sensitivity analysis and scanning laser Doppler vibrometer (SLDV) 
experiments further verify its robustness in real physical environments. The forward 
modeling paradigm established in this study possesses good theoretical portability and 
33.081026.98100.9971
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 3

provides a universal approach for the deep integration of physical priors with operator 
learning.
Keywords: ultrasonic guided wave, wavefield prediction, physics-informed neural 
operator, KL plate theory, defect scattering
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 4

1. Introduction
Plate-like structures are widely used in modern industrial fields represented by 
aerospace, shipbuilding, and petrochemical engineering, with their structural integrity 
directly linked to operational safety and economic efficiency. Owing to the advantages 
of long propagation distance, low attenuation, and high sensitivity to local thickness 
perturbations, ultrasonic guided wave (UGW) has emerged as a crucial information 
carrier in the nondestructive testing (NDT) and structural health monitoring (SHM) of 
these components [1, 2]. The interaction between guided waves and irregular defects 
within a plate induces complex scattering phenomena, including reflection, 
transmission, and diffraction [3]. The accurate solution of this full-field forward 
problem, involving multiple scattering and dispersion evolution dynamics, aims to 
establish a deterministic mapping from defect geometric distributions to dynamic 
wavefield responses. This task represents a fundamental physical challenge in 
understanding guided wave scattering mechanisms and serves as an indispensable 
computational foundation for subsequent quantitative inversion and high-resolution 
imaging [4, 5].
Existing research on guided wave propagation and scattering is generally 
categorized into three types: analytical, numerical, and data-driven methods [6]. 
Analytical methods are commonly based on Green’s function methods, Helmholtz 
integral equations, or generalized Bessel function expansions, are dedicated to 
mathematically deriving exact closed-form solutions for scattered wavefield [7, 8]. 
However, analytical solutions rely on regular geometric boundaries and simplified 
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 5

physical assumptions, making it difficult to directly handle full-wavefield scattering 
problems involving irregular defects [9, 10]. To overcome geometric topological 
constraints, numerical methods represented by the finite element method (FEM), finite 
difference method (FDM), and boundary element method (BEM) have become the 
mainstream standard for addressing guided wave propagation under complex boundary 
conditions [11, 12, 13]. By finely discretizing continuous partial differential equation 
(PDE) domains in space and time, numerical methods are capable of approximating 
complex physical boundaries and material inhomogeneities with high accuracy. 
Moreau et al. established a high-accuracy finite element scattering evolution model for 
irregular corrosion defects in plate structures [14]. Nguyen et al. utilized numerical 
algorithms to solve dynamic acoustic fields under different boundary conditions and 
analyzed their stability and accuracy [15]. Nevertheless, to ensure numerical stability 
for high-frequency wave motions, conventional numerical solvers still encounter 
significant computational efficiency bottlenecks in scenarios involving large-scale real-
time wavefield prediction or extensive forward simulations [16, 17].
To overcome the efficiency bottlenecks of iterative calculations in numerical 
methods, data-driven deep learning techniques have been introduced in recent years for 
forward prediction of physical fields [18, 19, 20]. Zhang et al. proposed a hybrid 
prediction model merging convolutional neural networks and Transformers to predict 
far-field scattering signals in thin plates with defects [21]. Wang et al. developed a 
fusion model based on compressive sensing and deep learning, which achieved high-
fidelity reconstruction of guided wave signals under sparse sampling conditions and 
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 6

successfully performed defect imaging using the reconstructed data [22, 23]. Although 
neural networks have demonstrated significant potential in fitting nonlinear mappings 
of complex systems, conventional purely data-driven models, due to their inherent 
black-box nature, not only depend heavily on massive amounts of high-quality training 
data, but also tend to generate non-physical artifacts in the absence of prior constraints 
[24, 25, 26]. Raissi et al. proposed physics-informed neural networks (PINNs), which 
substantially improved physical consistency by embedding PDEs into the loss function 
as regularization residuals [27]. Rasht-Behesht et al. combined PINNs with acoustic 
wave equations and successfully addressed seismic wave propagation and full 
waveform inversion (FWI) problems [28]. To break through the limitations of specific 
samples and grid resolutions, Lu et al. proposed the deep operator network (DeepONet) 
based on the universal operator approximation theorem, achieving continuous mapping 
in infinite-dimensional function spaces [29]. Subsequently, Wang et al. further 
proposed the physics-informed DeepONet, which achieved efficient learning of general 
solution operators for PDEs by combining automatic differentiation (AD) to calculate 
physical residuals [30]. In acoustics, Nair et al. combined the physics-informed operator 
network with non-rational B-splines geometric parameterization to realize the forward 
prediction of acoustic scattering fields for arbitrary scatterers [31]. Wang et al. 
demonstrated the effectiveness of physical priors through a series of progressive studies 
and proposed embedding discrete finite-difference operators into the network as an 
alternative to AD. This approach effectively circumvented computational memory 
bottlenecks and successfully predicted transcranial frequency-domain wavefield as well 
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 7

as S0 mode Lamb wave scattering in plate structures [32, 33, 34, 35]. Meanwhile, 
another class of architectures represented by the Fourier neural operator (FNO) has 
made breakthrough progress in complex spatiotemporal evolution simulations, owing 
to its advantage in capturing global correlations in the frequency-domain [36, 37].
Despite the widespread application of deep learning to wavefield problems, 
existing research on guided waves primarily focuses on inverse problems, with a 
noticeable scarcity of studies targeting the rapid forward prediction of full 
spatiotemporal wavefield under complex corrosion-like defects [38, 39]. Furthermore, 
most current models rely on simplified scalar acoustic wave equations, failing to 
effectively characterize the dispersion effects and local stiffness discontinuities 
inherent in plate waves [40]. Constrained by the computational bottlenecks associated 
with high-order AD during full-field training, a high-resolution spatiotemporal solution 
framework that simultaneously accounts for realistic dispersion evolution and complex 
geometric coupling under explicit physical constraints remains unavailable.
Against this background, this paper proposes the physics-informed deep U-Fourier 
neural network (PI-DeepUFNet), a novel neural operator framework for the fast 
prediction of full guided wavefield in plates with irregular defects. By incorporating 
the fourth-order dynamic equations from Kirchhoff-Love (KL) plate theory as physical 
priors, the proposed framework integrates the local geometric feature extraction 
capabilities of U-Net with the global spatiotemporal evolution modeling of 3D FNO. 
This synergy enables efficient and physics-consistent full spatiotemporal prediction of 
ultrasonic guided wavefield in variable thickness plates.
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 8

The main contributions of this work are as follows:
(1) KL plate theory is employed as an alternative to simplified acoustic models to 
construct a finite-difference forward operator for flexural waves under spatially 
variable thickness conditions, and a dataset containing different scattering 
characteristics is generated.
(2) The PI-DeepUFNet architecture is designed to integrate local geometric feature 
extraction with global spatiotemporal modeling, while explicitly embedding discrete 
physical operators to achieve the joint optimization of data and physics.
(3) The performance and engineering utility of the proposed method in complex 
irregular defect scenarios are systematically evaluated through numerical simulations, 
ablation studies, and experimental validation.
The remainder of this paper is organized as follows: Section 2 delineates the 
mathematical model of guided wave propagation based on KL plate theory and its 
dispersion characteristics. Section 3 introduces the discrete forward modeling method, 
the PI-DeepUFNet architecture, and the design of the physics-informed loss function. 
Section 4 analyzes the convergence characteristics of model training and wavefield 
prediction results, with the role of physical constraints evaluated via ablation 
experiments. Section 5 discusses the sensitivity of the model to various defect 
morphologies and validates its applicability in real-world scenarios through scanning 
laser Doppler vibrometer (SLDV) experiments. Section 6 concludes the paper.
2. Theory
This section derives the dynamic partial differential governing equation under 
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 9

spatially variable thickness conditions based on KL thin plate theory. This variable-
coefficient equation serves as the mathematical foundation for guided wave 
propagation in complex defective plates and will be embedded as prior physical 
knowledge into the loss function of the deep neural operator in subsequent sections.
2.1 Dispersion characteristics
KL theory describes the flexural behavior of isotropic thin plates under transverse 
loading, where the thickness is significantly smaller than the in-plane dimensions [41, 
42]. Consider a thin plate with a spatially varying thickness , where the 
undeformed mid-surface is set at  and the out-of-plane displacement is denoted 
by , as shown in Fig. 1(a). Based on the Kirchhoff hypothesis, normals to the 
mid-surface remain straight and perpendicular to it after deformation, and transverse 
shear deformation is negligible. Consequently, under small-deflection conditions, the 
in-plane strains induced by mid-surface curvature are linearly distributed along the 
thickness direction. Accordingly, the normal strains of the plate can be expressed as:
(2)
and the in-plane shear strain is:
(3)
It can be seen that the strains vary linearly with the distance from the mid-surface, 
reaching equal and opposite values at the top and bottom surfaces. This corresponds to 
the upper and lower layers being under tension and compression, respectively, during 
plate bending. Under the plane stress assumption, the isotropic linear elastic 
(,)hxy0z(,,)wxyt 2222(,,,),(,,,)xxyywwxyztzxyztzxy2(,,,)2xxwxyztzxy
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 10

constitutive relationship is given by:
(4)
where,  is Young's modulus,  is Poisson's ratio, and  is the shear modulus.
By integrating the stresses along the thickness direction with respect to the mid-
surface, the bending moments per unit width are defined as:
(5)
Introducing the local flexural rigidity of the plate:
(6)
the relations between bending moments and curvatures can then be expressed as:
(7)
Newton's second law governs the force equilibrium of the plate element. As shown 
in Fig. 1(b), the Kirchhoff shear forces are defined as:
(8)
Considering the inertial force and external load , the equilibrium equation for 
the micro-element can be written as:
(9)
where,  represents the material density. Substituting Eq. (7) yields the equilibrium 
equation in terms of moments:
22(),(),11xxxxyyxxyyxxxyxyEEGEG/2/2/2/2/2/2d,(,,)d,(,,)dhhhxxxxyyyyxyxyhhhMzzMxytzzMxytzz32(,)(,)12(1)EhxyDxy222222222(,,)(,)(,,)(,)(,,)(,)(1)xxyyxywwMxytDxyxywwMxytDxyyxwMxytDxyxy,xyyyxyxxxyMMMMQQxyyx(,,)qxyt22(,,)(,)yxQQwqxythxyxyt
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 11

(10)
Further substitution of Eq. (6) yields the KL dynamic governing equation for thin 
plates with variable thickness undergoing out-of-plane vibrations at small deflections. 
For general cases involving variable stiffness, the compact form can be expressed as:
(11)
where,  is the primary biharmonic term encompassing the low-order 
effects of stiffness variation, and  denotes the mixed operator term introduced 
by the stiffness gradient. The third term represents the inertial component. This 
equation explicitly demonstrates that irregular thickness variations introduce additional 
source terms into the equation, which constitutes the physical origin of scattering wave 
generation.
Fig. 1. Schematic of the KL thin plate model. (a) Geometric configuration and coordinate 
system, (b) Bending moments and Kirchhoff shear forces on a plate element.
222 22222(,,)(,)xyyyxxMMMwqxythxyxxyyt2222()(1)(,)wDwDwhqt22()Dw(,)Dw
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 12

2.2 Dispersion relation
To quantitatively analyze the influence of thickness variations on guided wave 
propagation, a local homogenization treatment is applied to the previously mentioned 
KL governing equations. By assuming the thickness  and flexural rigidity  are 
constant within a local element and neglecting external forces, the governing equation 
simplifies to:
(12)
Under the assumption of thin plates at low frequencies, this fourth-order partial 
differential equation accurately represents the asymptotic behavior of the  Lamb 
wave mode and inherently captures the dispersion characteristics from the underlying 
physical mechanism.
Substituting the plane harmonic wave solution  yields the 
dispersion equation:
(13)
where,  is the wavenumber and  is the angular frequency.
Equation (12) establishes a nonlinear relationship between angular frequency and 
wavenumber, specifically . Consequently, the phase velocity  and group 
velocity  can be derived as:
(14)
These expressions indicate that wave propagation speed increases with frequency. 
Specifically, higher frequency components travel faster, which leads to gradual pulse 
hD42420wwDhxtA0()(,)ikxtwxtAe420Dkhk2kpcgc,2pgDdDckckkhdkh
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 13

broadening and waveform distortion of broadband excitation signals during 
propagation, manifesting as clear dispersion effects.
Table 1
Material parameters.
(kg/m3) (GPa) (mm)
2700 70 0.33 3
The dispersion curves for phase and group velocities, generated by substituting the 
aluminum plate parameters from Table 1, are shown in Fig. 2. The normalized spectrum 
of the excitation signal is overlaid as a green dashed line to define the velocity range 
within the effective bandwidth.
Fig. 2. Theoretical dispersion curves of the  Lamb wave mode for a 3 mm thick aluminum 
plate based on KL theory.
In contrast to simplified second-order acoustic wave models, KL plate theory 
incorporates fourth-order spatial derivative terms to precisely capture dispersion effects 
arising from flexural rigidity. Since the broadband propagation of guided waves in thin-
walled media is inevitably accompanied by pulse broadening and distortion, non-
dispersive models that ignore these effects are inadequate for accurate full-wavefield 
prediction in complex spatiotemporal domains. This derivation clarifies the dispersion 
Eh
A0
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 14

laws of guided waves in plates and underscores the necessity of integrating this physical 
mechanism into wavefield prediction models.
3. Methodology
The previous section established the governing equations for flexural guided 
waves in plates with variable thickness based on linear elastodynamics and KL plate 
theory. This section further defines the wavefield solution problem under conditions of 
irregular defects, introduces the numerical forward operator and the PI-DeepUFNet 
framework employed for the solution.
3.1 Problem Formulation and Numerical Forward Operator
For thin plate structures containing irregular defects, the propagation of  
mode Lamb waves can be regarded as a scattering problem described by a variable-
coefficient biharmonic equation. Let  be the two-dimensional spatial domain 
of the plate, where  denotes the spatial coordinates. Irregular defects are 
modeled geometrically through a thickness field  that varies continuously in 
space. Non-uniform variations in thickness at the defect locations simultaneously 
induce changes in local mass distribution and flexural rigidity, thereby acting as 
equivalent secondary scattering sources during the propagation of guided waves.
Within this physical framework, the total out-of-plane displacement wavefield 
 can be decomposed into the superposition of the incident and scattered 
wavefield:
(15)
where,  represents the reference incident wavefield propagating in a uniform thin 
A02R(,)xyx()hx(,)wtx(,)(,)(,)incscatwtwtwtxxxincw
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 15

plate without defects, and  is the scattered wavefield excited by the thickness 
perturbation . The primary objective of this study is to construct a nonlinear 
mapping operator  that, through optimization of parameters , establishes an end-
to-end mapping from the two-dimensional geometric distribution of defects to the three-
dimensional full spatiotemporal wavefield:
(16)
Training the deep neural operator  requires the construction of a high-quality 
dataset containing diverse irregular defects and complex scattering features. Since 
traditional analytical solutions are typically restricted to simple boundaries, such as 
circles or rectangles, they are inadequate for handling natural corrosion defects with 
arbitrary shapes. Consequently, this study adopts the FDM to build a high-fidelity 
numerical forward operator [12]. Considering that the KL governing equation involves 
complex spatial derivatives of the fourth-order, direct discretization of the fourth-order 
operator would lead to an excessively large calculation template and reduced numerical 
stability. To address this, we rewrite the original partial differential equation into a 
coupled form of second-order operators for bending moment calculation and dynamic 
equilibrium. At a discrete time step , the bending moment field  
corresponding to the spatial displacement field  is calculated as:
(17)
where,  is the distribution of flexural rigidity determined from the thickness field 
,  denotes the Hadamard product, and  represents the convolution operation. 
 is the second-order spatial difference operator, and the subscript  indicates a 
scatw()hxG0ˆ(,)((),(,))wthwtxxxGG
nt ()nxM()nwx()()()nnDwxxLxM()Dx()hxL ()
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 16

linear combination that accounts for the Poisson's ratio coupling term. By utilizing the 
spatial divergence of the bending moment field, the current out-of-plane acceleration 
field  can be further determined:
(18)
Equations (16) and (17) facilitate the construction of a complete discrete 
differential operator, that describes the transient dynamic behavior of flexural guided 
waves in plates with variable thickness. In the temporal direction, explicit time 
integration is performed using a central difference scheme. The wavefield evolution 
equation is expressed as:
(19)
where,  is the external excitation source term in discrete form. This explicit finite 
difference time-domain (FDTD) scheme offers advantages including algorithmic 
simplicity and high efficiency in parallel computing. With appropriate grid partitioning, 
it precisely captures subtle scattering details of guided waves induced by irregular 
thickness variations. To avoid non-physical numerical dispersion and ensure the 
convergence of the explicit integration, the time step  and the spatial grid spacing 
 must strictly satisfy the Courant-Friedrichs-Lewy (CFL) stability constraint. This 
ensures that the physical propagation distance of the wave within a single time step 
does not exceed the spatial grid spacing [17].
Provided that the initial and boundary conditions along with material parameters 
are fixed, the complete spatiotemporal evolution of the wavefield is uniquely 
determined by the geometric thickness distribution of the defects . The numerical 
()nax1()()()nnahxLxxM1122(,)nnnnnKLwwwtwhSKnStx()hx
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 17

forward physical model constructed by the FDTD scheme can be formalized as a 
deterministic nonlinear operator:
(20)
This operator provides accurate supervisory signals for the subsequent learning of 
the deep neural operator. By performing parallel forward calculations on multiple sets 
of randomly generated irregular thickness fields, a guided wave wavefield dataset rich 
in diverse scattering patterns is efficiently produced. This dataset establishes the 
foundation for the training and validation of the PI-DeepUFNet framework.
3.2 Forward modeling of helical wave propagation
To address the computational efficiency and accuracy bottlenecks inherent in the 
forward modeling of guided waves in plates with variable thickness, we develop the 
physics-informed deep U-Fourier neural network. This framework integrates the 
advantages of U-Net in local geometric feature extraction across multiple scales with 
the capacity of 3D FNO for modeling global spatiotemporal evolution. This synergy 
enables an efficient approximation of the nonlinear operator defined in Eq. (19) and 
establishes a direct mapping from defect thickness distributions to full spatiotemporal 
wavefield.
()hwF
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 18

Fig. 3. Schematic of the proposed PI-DeepUFNet architecture. The framework takes the defect 
thickness map and initial wavefield as inputs. A U-Net encoder extracts local multi-scale 
geometric features, while a 3D FNO propagator models the global spatiotemporal wave 
evolution. The discrete KL plate equations are incorporated into the loss function to enforce 
dynamical consistency.
The overall architecture of the network is illustrated in Fig. 3. The input to PI-
DeepUFNet is a concatenated tensor comprising the thickness field  and the initial 
wavefield . This input is first processed by a geometric feature encoder based 
on the U-Net architecture, which extracts local geometric features and multi-scale 
spatial variation information through downsampling and skip connection mechanisms. 
The low dimensional physical space features encoded by U-Net are mapped into high 
dimensional latent features . Subsequently, these features are expanded along the 
temporal dimension via a temporal extension module, while a coordinate encoder is 
introduced to encode the full spatiotemporal grid coordinates . Following 
feature concatenation, a lifting layer projects the mixed features into a higher 
dimensional channel space, completing the embedding from the spatial domain into the 
spatiotemporal manifold.
The 3D FNO propagator is the core module of PI-DeepUFNet, consisting of 
multiple stacked Fourier operator blocks designed to learn the global propagation 
()hx0(,)wtx()vx(,,)xyt
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 19

operator of waves within the medium. Unlike conventional convolutional neural 
networks that are limited by local receptive fields, each layer in this module 
incorporates a frequency-domain global convolution branch and a spatial-domain local 
linear branch. For the hidden feature  at the  layer, the transition to the  
layer is expressed as:
(21)
where,  and  denote the Fourier transform and its inverse, respectively. 
 represents a learnable complex weight tensor in the frequency domain,  
denotes the local linear convolution weights in the spatial domain, and  is the 
GELU nonlinear activation function. The frequency-domain branch efficiently captures 
long-range dependencies in guided wave propagation and the dispersion characteristics 
induced by flexural waves, while the spatial-domain branch preserves local high-
frequency information and fine structures. These two components are coupled in a 
residual form to ensure stable modeling of the spatiotemporal evolution of guided 
waves.
Following propagation through multiple FNO layers, the resulting high 
dimensional features are mapped back to the physical space via a projection decoder, 
yielding the predicted out-of-plane wavefield . To ensure that the predictions 
approximate the numerical forward modeling data while strictly satisfying the 
dynamical laws of guided wave propagation, a compound loss function is designed. 
This function comprises a data-driven term  and a physics-informed term :
(22)
(,)lvtxl 1l11(,)(,)()(,)lllllvtWvtRvxxkFFFk1FlR lW
ˆ(,)wtxdataLphyL12totaldataphyLLL
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 20

where,  and  are weight hyperparameters used to balance the gradients of the 
two loss components. Through this dual data and physics-driven training approach, PI-
DeepUFNet maintains high predictive precision while effectively suppressing the 
generation of non-physical solutions. This achieves efficient and reliable prediction of 
full spatiotemporal guided wavefield under conditions involving complex irregular 
defects.
3.3 Physics-informed loss function
Although PI-DeepUFNet possesses strong high-dimensional feature fitting 
capabilities, training exclusively on limited supervised data can still lead to overfitting 
or the generation of results that violate physical laws, such as abnormal wave speed 
dispersion or non-physical scattering artifacts. The physics-informed loss term  is 
a critical design in this study, the core idea of which is to directly embed the KL control 
equations into the computational graph as discrete finite-difference operators, thereby 
imposing dynamical consistency constraints on the predicted wavefield. Unlike 
traditional PINNs that rely heavily on AD to calculate high-order spatiotemporal 
derivatives, we convert the previously constructed discrete difference operators into 
non-learnable deterministic convolutional layers to participate in the forward 
propagation of the network, which accommodates high-resolution full-field grid data 
while significantly reducing memory overhead.
For the predicted wavefield sequence , the predicted bending moment 
distribution  is first computed using the bending moment operator from Eq. 
(16), which subsequently yields the discrete acceleration field:
12
phyL
ˆ(,)wtx()nxM
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 21

(23)
Based on the requirements of the explicit second-order central difference time 
integration scheme, any physically consistent dynamical evolution must satisfy strict 
recurrence constraints across any three consecutive time steps. Thus, the physical 
residual  is defined as the deviation between the predicted wavefield at the 
central time step and the theoretical value derived from the discrete dynamical 
equations:
(24)
The final physics loss function is defined as the mean squared error of the residuals 
across all interior time steps:
(25)
By minimizing , the network is explicitly constrained within a solution space 
that conforms to the physical laws of guided wave propagation. This prior constraint 
not only significantly improves predictive accuracy in complex scattering regions and 
effectively suppresses non-physical predictions, but also fundamentally ensures the 
dynamical consistency and generalization capability of the model, allowing it to 
maintain highly stable and reliable forward performance even when encountering 
completely new and unknown defect topologies.
4. Numerical experiments and results
This section evaluates and analyzes the performance of the proposed method 
through numerical experiments. It details the dataset generation and processing, 
1()*()()nnahxLxxMˆ(,)whR1121ˆˆˆˆ(,)2nnnnwhwwwtaR2211ˆ()(,)intNphyntintwhNT‖‖TLRphyL
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 22

demonstrates the training convergence characteristics and prediction results, and 
quantifies the specific role of physical constraints in enhancing prediction accuracy 
through ablation experiments.
4.1 Dataset generation
To train and validate the PI-DeepUFNet architecture, a wavefield dataset 
encompassing diverse multi-scale and multi-morphology defect features was generated 
using the numerical forward operator defined in Eq. (19). The study focuses on a 
uniform aluminum plate, the material parameters of which are detailed in Table 1. The 
region of interest (ROI) for the numerical model is . To eliminate 
interference from boundary reflections on the forward scattered field, the actual 
simulation domain extends beyond this region and is equipped with complete absorbing 
boundaries. To simulate irregular corrosion in engineering scenarios, thickness loss 
zones were randomly generated within a central 120 mm diameter area based on cloud-
scan maps of actual corrosion surfaces. The maximum wall loss for each sample ranges 
from 0% to 50%, with equivalent diameters between  and . The 3D 
physical layout of the simulation model is illustrated in Fig. 4(a).
Ultrasonic guided waves are excited by a vertical point force distribution applied 
to the top surface of the plate, with its relative coordinates in the ROI set at (250 mm, 
50 mm). The excitation signal consists of a 5-cycle sine pulse modulated by a Hann 
window at a center frequency of 200 kHz. In numerical calculations, the spatial step is 
set to , discretizing the ROI into a  grid. To satisfy the CFL 
stability condition, the time step is set to , with a total simulation duration 
500mm500mm10mm100mm1mmdx5005000.01sdt
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 23

of . Fig. 4(b) displays the thickness distribution and a total wavefield snapshot 
at  for a typical sample, clearly illustrating the complex scattering 
phenomena induced by irregular thickness variations.
Given the grid-independent advantages of neural operator architectures, the raw 
high-dimensional dataset was downsampled to accelerate model training while 
preserving core wavefield dynamics. Specifically, the spatial resolution was reduced to 
 and the time step was extracted at intervals of . As shown in Fig. 4(c), 
the downsampled signal aligns closely with the original time-domain waveform, 
introducing no significant numerical dissipation or phase distortion. Additionally, 
based on the group velocity of the  mode, a core time interval from  to 
 was selected as the effective observation window for the scattering evolution. 
After processing, the input for each sample consists of a  tensor 
containing the 2D thickness distribution and the initial wavefield, while the 
corresponding target labels are full spatiotemporal wavefield tensors of size 
.
Fig. 4. Data generation and preprocessing. (a) Schematic layout of the 3D computational 
domain, (b) thickness distribution and full-wavefield snapshot at  for a 
representative sample, (c) comparison of time-domain waveforms before and after 
downsampling.
Based on the above process, a total of 2000 independent samples were generated 
200s100st2mm0.5sA025s125s 2502502201250250 100ust
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 24

and randomly split into training, validation, and test sets at a ratio of 8:1:1. It is 
emphasized that the thickness distributions in the test set are entirely independent of 
the training data and contain novel geometric topologies never encountered during 
training. This ensures a strict evaluation of the generalization and dynamical 
extrapolation capability of PI-DeepUFNet under unknown geometric conditions.
4.2 Wavefield Prediction Results
The training and inference processes in this study were conducted on a workstation 
equipped with an NVIDIA RTX 4090 GPU. The Adam optimizer was employed during 
training with an initial learning rate of 0.001 and a total of 150 epochs. To effectively 
balance absolute fitting accuracy with dynamical consistency, a hybrid loss function as 
defined in Eq. (21) was adopted. To prevent gradient divergence caused by excessive 
physical constraints in the early stages of training, a dynamic weight preheating strategy 
was implemented. Specifically, for the first 10 epochs, the data loss weight was set to 
 and the physics loss weight to  for preliminary mapping based on pure 
data. Subsequently, both weights were adjusted to 0.5 for the remainder of the training. 
The optimal model was saved by monitoring the minimum fitting loss on the validation 
set, and all subsequent performance evaluations were conducted based on this optimal 
model.
1120
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 25

Fig. 5. Loss curves for the PI-DeepUFNet model and the baseline DeepUFNet model on the 
training and validation sets. (a) Data loss curves, (b) physics loss curves.
Fig. 5 compares the loss decline curves of the model on the training and validation 
sets. The data loss reflects the absolute fitting performance of the model relative to the 
ground truth wavefield, while the physics loss quantifies the degree to which the 
predicted wavefield adhere to physical propagation laws. Both data and physics losses 
decrease rapidly during the initial training phase, after which the data loss curve tends 
to plateau. Around epoch 37, the model escapes the local optimum due to changes in 
weighting and the learning rate. As training progresses, the data loss declines steadily, 
eventually converging to  on the validation set and  on the 
training set. The evolution of the physics loss exhibits more complex nonlinear 
dynamics, undergoing an upward jump before continuing to converge. This suggests 
that while data fitting and physical constraints point toward consistent optima in a 
global direction, their gradient descent spaces do not completely overlap, reflecting the 
process of the network seeking an optimal balance between the two. Final convergence 
values reached  on the validation set and  on the training set. 
Overall, the training and validation curves maintain high consistency with no 
significant overfitting, effectively validating the role of the hybrid loss design in 
51.511051.3510
75.031075.1210
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 26

enhancing generalization.
Fig. 6. Thickness distributions of four randomly selected defect samples from the test set. (a) 
Sample #6, (b) sample #3, (c) sample #7, (d) sample #56.
To further evaluate the ability of the model to predict scattering fields from 
complex defects, four representative defect samples were randomly selected from the 
test set, with their thickness distributions shown in Fig. 6. Taking sample #6 as an 
example, Fig. 7 provides a detailed comparison of full-field evolution snapshots at 
, , , and , including the FDM numerical ground truth, model 
predictions, and their residuals. The comparison shows that the predicted wavefront 
shape, spatial position, and amplitude distribution almost perfectly overlaps with the 
high-fidelity simulation results. At , as the incident wave begins to interact with 
the defect region and the wavefront undergoes local distortion, the model accurately 
captures the changes in wave speed and wavefront perturbation features near the defect. 
By , the guided wave gradually passes through the defect region, and the model 
successfully predicts the generation of scattered waves and the formation of multiple 
scattering ripples. As time progresses to  and , distinct scattered waves 
and interference structures appear, and the model continues to reconstruct the complex 
wavefront structures and amplitude distributions with high-fidelity. As shown in the 
residual maps in Fig. 7(c), errors are primarily concentrated in the strong scattering 
regions near the defects, with peak errors not exceeding 13% and the global average 
60s80s100s120s60s80s100s120s
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 27

error remaining stable below 1%. This intuitively demonstrates that the neural operator 
possesses exceptionally high prediction fidelity across the entire field.
Fig. 7. Wavefield snapshots for defect sample #6 at different time steps. The rows from top to 
bottom correspond to the FDM ground truth, PI-DeepUFNet predictions, and absolute residuals, 
respectively.
To quantitatively evaluate the performance of the proposed PI-DeepUFNet, we 
employ three standard metrics with the relative L2 error (RL2) to measure overall 
global accuracy, the root mean square error (RMSE) for absolute deviation, and the 
Pearson correlation coefficient (PCC) to assess waveform phase consistency and shape 
similarity. Detailed calculation results are listed in Table 2, where sample #6 achieved 
RMSE, RL2, and PCC values of , , and , respectively. 
Comprehensive statistics for the first 100 samples in the test set yield an average RMSE 
of , an average RL2 of , and an average PCC as high as . 
This near-unity PCC indicates that the predicted waveforms are highly consistent with 
the labels in terms of evolution trends and amplitude variations, proving from a physical 
31.971024.46100.999033.081026.98100.9971
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 28

perspective that the model has accurately internalized the complex dispersion attributes 
of guided waves. This capacity for precise capture of phase features enables the model 
to reliably identify subtle scattering ripples caused by irregular defects, thereby laying 
a solid data foundation for high-resolution imaging and quantitative evaluation based 
on time of flight.
Table 2
Quantitative evaluation of the wavefield predictions generated by PI-DeepUFNet.
Sample PI-DeepUFNet
RMSE RL2 PCC
Test set
# 3 5.20×10-3 1.18×10-1 0.9931
# 6 1.97×10-3 4.46×10-2 0.9990
# 7 4.52×10-3 1.02×10-1 0.9948
# 56 8.45×10-4 1.92×10-2 0.9998
Avg (0~100) 3.08×10-3 6.98×10-2 0.9971
Experiment
# 1 1.73×10-3 3.93×10-2 0.9992
4.3 Ablation Study: Impact of physics-informed constraint
To quantitatively evaluate the specific contribution of physics constraints, an 
ablation study was conducted. A purely data-driven neural network without the PDE 
residual loss term is defined as the DeepUFNet baseline. During training of the baseline 
model, the PDE residual serves only as a monitoring metric and does not participate in 
backpropagation. As shown in the loss curves of Fig. 5, the convergence of the model 
without physics constraints is significantly slower, with final data and physics losses 
stabilizing at approximately 1.7 and 2.8 times those of the PI-DeepUFNet model 
respectively. Notably, after a jump in the data loss curve of the baseline model near 
epoch 75, the physics loss fails to return to its previous low point despite a continued 
smooth decline in data loss. This suggests that while the purely data-driven model 
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 29

achieves acceptable fitting within the training distribution, its mapping relationship 
does not strictly follow the underlying wave dynamics, leading to energy divergence or 
phase misalignment during extrapolation. In contrast, the physics guided mechanism 
constrains the network to converge toward optimal solutions that strictly satisfy the KL 
dynamical equations within the hypothesis space, demonstrating superior convergence 
stability and energy conservation.
Fig. 8. Time domain waveforms at observation points for sample #6 including the FDM 
benchmark, PI-DeepUFNet predictions, and the baseline DeepUFNet results.
To verify this conclusion regarding local details, seven observation points were 
uniformly selected along the forward scattering directions from 150° to 210° in the 
shadow region behind the defect of sample #6 as shown in Fig. 6(a). As illustrated in 
Fig. 8, the black solid lines represent the FDM simulation benchmark, while the blue 
and red dashed lines represent the predictions of PI-DeepUFNet and the baseline model 
respectively. The results show that the waveforms predicted by PI-DeepUFNet overlap 
almost perfectly with the FDM benchmark, maintaining high consistency in wave 
packet phase, amplitude, and interference patterns. However, the baseline model 
without physics constraints exhibits varying degrees of distortion as the observation 
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 30

angle changes. This error is significantly concentrated in the middle and late stages of 
wave packet evolution. The physical root of this discrepancy lies in the failure of the 
purely data-driven network to fully internalize the coupling between spatial thickness 
distribution and guided wave dispersion characteristics, resulting in numerical 
dissipation and phase shifts during complex transmission and diffraction processes.
To further compare the effect of physics constraints across the entire spatial 
domain, the scattered wavefield of four test samples at  were extracted. To 
highlight the scattering features caused by defects by eliminating the masking effect of 
the strong background incident field, the scattered wavefield  is defined as:
(26)
where,  is the reference wavefield in a uniform plate without defects.
The comparison results in Fig. 9 indicate that since corrosion defects belong to 
typical thickness reduction geometries, their overall reflection response is weak and the 
wavefield is dominated by transmission and forward scattering behavior. For sample 
#6, both models reproduce the scattered wavefield distribution relatively well, but PI-
DeepUFNet offers higher fidelity with PCC values reaching 0.9954 and 0.9878 
respectively. For sample #3, which features severe thickness reduction and complex 
scattering, the wavefield predicted by the baseline model shows visible structural 
distortion, with the PCC value dropping from 0.9576 to 0.8988. For the large defect in 
sample #7 and the small defect in sample #56, the PI-DeepUFNet predictions also 
remain closer to the FDM overall. Particularly in sample #56, DeepUFNet shows 
significant local distortion while PI-DeepUFNet maintains good reconstruction of the 
80sscatw(,)(,)(,)scattotalbasewtwtwtxxxbasew
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 31

scattered field structure. Corresponding PCC values improved from 0.9274 and 0.8626 
to 0.9720 and 0.9105.
Fig. 9. Scattered wavefield for various defect samples at , where rows from top to 
bottom correspond to the FDM numerical benchmark, PI-DeepUFNet predictions, and the 
baseline DeepUFNet results.
Beyond the qualitative comparison of waveforms and phase indicative PCC, we 
further calculated RMSE and RL2 to quantitatively evaluate the scattered field 
predictions for the first 100 samples in the test set at  as listed in Table 3. Results 
indicate that PI-DeepUFNet outperforms the purely data-driven baseline model across 
all representative samples and in terms of the overall average. The average results for 
the test set show that the RMSE, RL2, and PCC for PI-DeepUFNet are , 
, and  respectively, while the corresponding metrics for 
DeepUFNet are , , and .
In summary, the systematic ablation experiments and multi-dimensional 
quantitative evaluations confirm the scientific validity of embedding the KL dynamical 
80st80s34.031012.59100.958936.281013.74100.9191
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 32

partial differential operator into the neural network. Guided by prior physical 
knowledge, the neural network is no longer limited to fitting shallow data mapping 
relationships but learns the physical laws of guided wave propagation and dispersion at 
the mechanistic level. This deep integration of physics and data fundamentally 
suppresses the non-physical artifacts common in black box deep learning models. It not 
only grants the model exceptional generalization capability when facing unknown 
complex defects but also establishes a reliable physical foundation for high resolution 
ultrasonic NDT and imaging based on deep learning.
Table 3
Quantitative evaluation of the predicted scattered wavefield at  for the proposed PI-
DeepUFNet and the baseline DeepUFNet.
Sample PI-DeepUFNet DeepUFNet (Baseline)
RMSE RL2 PCC RMSE RL2 PCC
Test set
# 3 6.91×10-3 2.89×10-1 0.9576 1.05×10-2 4.41×10-1 0.8988
# 6 2.54×10-3 9.90×10-2 0.9954 4.18×10-3 1.63×10-1 0.9878
# 7 5.94×10-3 2.35×10-1 0.9720 9.46×10-3 3.75×10-1 0.9274
# 56 7.51×10-4 4.16×10-1 0.9105 9.31×10-3 5.16×10-1 0.8626
Avg 
(0~100)
4.03×10-3 2.59×10-1 0.9589 6.28×10-3 3.74×10-1 0.9191
5. Discussion
This section further discusses the predictive performance of the PI-DeepUFNet 
model with a focus on evaluating how geometric features such as defect size and depth 
influence model accuracy. Additionally, scanning experiments are designed to validate 
the actual representation capability of the model.
5.1 Sensitivity to defect morphology
In practical ultrasonic NDT, the geometric topology of corrosion defects exhibits 
a high degree of randomness. As described by the Kirchhoff-Helmholtz integral 
80st
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 33

equation, the distribution characteristics of the forward scattered field depend strictly 
on the geometric shape of the scattering boundary and the perturbation of local flexural 
rigidity:
(27)
where,  represents the scattered field,  is the Green’s function, and  is the 
incident field. To systematically investigate the sensitivity of the PI-DeepUFNet model 
to geometric features, we generated an additional independent test set containing 500 
irregular defect samples and calculated various evaluation metrics for the scattered field 
predicted at .
Fig. 10. Impact of defect size on PI-DeepUFNet prediction accuracy.
Fig. 10 intuitively reflects the impact of equivalent defect diameter on the 
predicted RMSE. The horizontal axis represents the defect size relative to the center 
frequency wavelength  ranging from  to . It can be observed that 
when the defect size is small, its acoustic response resembles point scattering and the 
wavefield distortion is relatively simple. The model exhibits extremely low prediction 
errors with RMSE concentrated around . As the defect size increases, the 
()()()()()()()incsc incSugugudSnnrrRrRRrrrscugincu80s
1.077.993210
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 34

prediction error shows a slow upward trend due to the increase in scattering boundary 
length and the intensification of internal multi-interference effects. After the defect size 
exceeds approximately , the degradation of accuracy due to defect size gradually 
plateaus. The RMSE no longer continues to diverge but fluctuates within a narrow 
range around . This phenomenon indicates that PI-DeepUFNet possesses 
excellent representation capability for large defects. Its global FNO has successfully 
learned macro-scattering physical patterns and does not suffer from significant 
performance degradation as geometric scales increase.
Compared to the lateral size, the longitudinal depth of the defect has a more direct 
nonlinear impact on the local wave speed and energy evolution of guided waves. Fig. 
11 illustrates the relationship between wall loss and predicted RMSE which exhibits 
clear piecewise evolution characteristics. When the thickness loss is below 21%, the 
perturbation of local stiffness is weak and the model exhibits extreme robustness with 
RMSE stably distributed at a low level of  and not deteriorating with increasing 
depth. In the 21% to 28% range, RMSE enters a transition zone with an upward trend 
as local frequency dispersion effects intensify. When the wall loss further exceeds 28% 
up to 50%, the overall error center continues to rise but the distribution of data points 
shows strong discreteness. This suggests that in the deep corrosion stage, wall loss is 
no longer the sole dominant factor controlling prediction error, as high-order geometric 
features such as complex edge contours of the defect begin to significantly interfere 
with the scattering process.
2.53410
3110
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 35

Fig. 11. Impact of maximum wall loss on PI-DeepUFNet prediction accuracy.
Overall, although the wavefield prediction accuracy of PI-DeepUFNet is 
influenced to some extent by physical parameters such as defect size and depth, its 
comprehensive performance remains at a high-fidelity level. In the test set of 500 
independent samples, the average RMSE of the scattered wavefield at  is only 
 and the average RL2 error is . Notably, the PCC representing 
waveform phase accuracy is distributed between 0.9751 and 0.9997 with an average of 
0.9947 and 88.2% of the samples exceeding 0.99. These statistical results indicate that 
with the help of deep guidance from physics differential equations, PI-DeepUFNet not 
only accurately maps the evolution representation of the spatiotemporal wavefield but 
also internalizes the underlying mechanical mechanisms governing guided wave 
propagation and scattering within a broad geometric parameter space. This establishes 
a solid physical foundation for quantitative health monitoring of thin-walled structures.
5.2 Experimental validation
To further validate the effectiveness of the proposed PI-DeepUFNet model in a 
real physical environment, an ultrasonic wave measurement platform was established 
80s33.811029.5110
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 36

based on a SLDV. The specimen is a  uniform aluminum plate with 
material and mechanical parameters identical to those used in the numerical simulation. 
To avoid interference from boundary reflections, a  area in the center of 
the plate was designated as the effective observation region. A piezoelectric (PZT) 
interdigital transducer was bonded at coordinates of  using epoxy 
resin to serve as the guided wave excitation source. An observation circle with a radius 
of 127 mm was centered on the research area to capture wavefield characteristics as 
shown in Fig. 12(a).
Fig. 12. (a) Schematic of the sample thickness map and observation circle setup, (b) FDM 
benchmarks for total and scattered wavefield at , (c) PI-DeepUFNet prediction results.
Before introducing experimental data, the total wavefield and scattered wavefield 
of PI-DeepUFNet and the FDM benchmark at  were first compared. As shown 
in the visualization results of Fig. 12(b) and Fig. 12(c), PI-DeepUFNet maintains high 
consistency with the FDM benchmark for both the main wavefront in the total 
wavefield and the local perturbations in the scattered wavefield. For this sample, the 
RMSE, RL2, and PCC of the total wavefield are , , and 0.9986 
respectively, while the corresponding metrics for the scattered wavefield are 
100010003mm500500mm(250mm,50mm)
80s80s32.101025.2410
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 37

, , and 0.9942.
During the actual SLDV measurement phase, an irregular defect with a topology 
identical to the simulation model was machined on the aluminum plate. Considering 
machining feasibility, the defect was depth approximated using a step-milling process. 
The experiment utilized a 5-cycle sine pulse at 200 kHz and 360 observation points 
were uniformly distributed along the observation circle. The lowest point of the circle 
is defined as 0° and signals were collected counterclockwise at each observation point. 
The overall experimental setup is illustrated in Fig. 13(a). The first arrival wave packet 
was extracted for each observation point and the peak-to-peak amplitude was calculated 
as the response magnitude for that direction. All observed values were then normalized. 
To ensure comparability among the three, an identical signal processing flow was 
applied to the FDM benchmark and PI-DeepUFNet prediction results to extract the 
angular magnitude distribution along the corresponding circle.
Fig. 13. Experimental setup and results. (a) SLDV experimental platform, (b) Comparison of 
normalized amplitude directivity patterns from FDM simulation, PI-DeepUFNet prediction, 
and SLDV measurement.
The polar plot in Fig. 13(b) characterizes the directional distribution of wavefield 
32.131011.0710
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 38

energy in space. Red circles represent the FDM benchmark, blue squares represent the 
PI-DeepUFNet prediction, and green triangles represent the actual SLDV measurement 
results. It can be observed that the three results exhibit strong correlation in overall 
distribution trends. In the main lobe direction near 0°, all three exhibit the highest 
response. The magnitude gradually decays as the angle deviates from the main 
propagation direction and a local peak caused by backscattering from the defect appears 
near 185°. This indicates that PI-DeepUFNet can accurately capture the main 
mechanisms governing the angular scattering characteristics of irregular defects. 
Further comparison reveals that PI-DeepUFNet and FDM almost overlap across most 
angles, suggesting that the model has accurately inherited the angular scattering 
characteristics of the numerical forward model. In contrast, although the overall trend 
of the SLDV measurement results is consistent, fluctuations in local peak values and 
positions occur within the 0° to 90° range and near 185°. These errors primarily stem 
from machining tolerances of the defect, coupling layer effects of the PZT, and 
fluctuations in the laser reflection signal under non-steady environments. Nevertheless, 
the high degree of consistency in the overall envelope fully validates the predictability 
of the model under real and complex physical boundaries.
In summary, PI-DeepUFNet demonstrates stable cross-domain prediction 
capability. Compared to traditional numerical simulations that are extremely time 
consuming, this physics guided model can complete the forward inference of 
spatiotemporal wavefield for complex irregular defects within seconds. This allows it 
to serve as an efficient real-time forward operator embedded into guided wave based 
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 39

intelligent quantitative inversion and imaging algorithms. It can also provide a reliable 
calculation engine for SHM digital twin wavefield libraries.
6. Conclusion
This study proposes a PI-DeepUFNet framework merging KL plate theory with 
physics-informed neural operators to address the rapid forward prediction of full 
spatiotemporal wavefield in plates containing irregular defects. By employing the 
fourth-order flexural wave dynamic equations under spatially variable thickness as 
physical priors, the proposed method integrates the local geometric feature extraction 
capabilities of U-Net with the global spatiotemporal evolution modeling of 3D FNO. 
This synergy establishes a robust mapping from defect thickness distributions to the 
resulting guided wave responses.
Quantitative results demonstrate that PI-DeepUFNet can accurately predict guided 
wave wavefield under complex defect conditions. Within the test set, the average 
RMSE, RL2, and PCC for wavefield predictions on unknown defect samples reached 
,  and 0.9971 respectively. Ablation analysis reveals that the 
embedding of physical constraints addresses the generalization bottlenecks of purely 
data-driven models, enhancing prediction accuracy and stability in complex scattering. 
Sensitivity analysis regarding defect size and depth features, along with cross validation 
through SLDV scanning experiments, further confirms the robustness and applicability 
of the model in real physical scenarios.
Compared to traditional numerical methods, this framework significantly 
improves forward modeling efficiency while maintaining high-fidelity, providing a 
33.081026.9810
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 40

computational foundation for SHM digital twins and subsequent quantitative inversion 
based on guided waves. Furthermore, the forward modeling paradigm established in 
this work possesses excellent theoretical portability. By replacing the discrete physical 
operators in the loss function with the governing equations of other target systems, this 
scheme can be extended to other forward modeling tasks involving diverse physical 
fields. This strategy provides a universal approach for the integration of physical priors 
with operator learning.
ACKNOWLEDGEMENTS
This work was supported in part by the National Natural Science Foundation of 
China under contract number 12474459, and in part by the Tianjin National Key 
Laboratory Major Project under contract number 24ZXZSSS00290.
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 41

REFERENCES
[1] Z. Su, L. Ye, Y. Lu, Guided Lamb waves for identification of damage in composite 
structures: A review, Journal of sound and vibration, 295 (2006) 753-780.
[2] M. Mitra, S. Gopalakrishnan, Guided wave based structural health monitoring: A 
review, Smart Materials and Structures, 25 (2016) 053001.
[3] P. Huthwaite, Guided wave tomography with an improved scattering model, 
Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 
472 (2016).
[4] J. Rao, M. Ratassepp, Z. Fan, Guided wave tomography based on full waveform 
inversion, IEEE transactions on ultrasonics, ferroelectrics, and frequency control, 63 
(2016) 737-745.
[5] P. Huthwaite, F. Simonetti, High-resolution guided wave tomography, Wave 
Motion, 50 (2013) 979-993.
[6] C. Willberg, S. Duczek, J.M. Vivar-Perez, Z.A. Ahmad, Simulation methods for 
guided wave-based structural health monitoring: a review, Applied Mechanics Reviews, 
67 (2015) 010803.
[7] O. Diligent, T. Grahn, A. Boström, P. Cawley, M.J. Lowe, The low-frequency 
reflection and scattering of the S 0 Lamb mode from a circular through-thickness hole 
in a plate: Finite element, analytical and experimental studies, The Journal of the 
Acoustical Society of America, 112 (2002) 2589-2601.
[8] T. Grahn, Lamb wave scattering from a circular partly through-thickness hole in a 
plate, Wave motion, 37 (2003) 63-80.
[9] B. Poddar, V. Giurgiutiu, Scattering of Lamb waves from a discontinuity: An 
improved analytical approach, Wave Motion, 65 (2016) 79-91.
[10]L. Moreau, M. Caleap, A. Velichko, P. Wilcox, Scattering of guided waves by 
through-thickness cavities with irregular shapes, Wave Motion, 48 (2011) 586-602.
[11]F. Moser, L.J. Jacobs, J. Qu, Modeling elastic wave propagation in waveguides 
with the finite element method, Ndt & E International, 32 (1999) 225-234.
[12]R. Balasubramanyam, D. Quinney, R. Challis, C. Todd, A finite-difference 
simulation of ultrasonic Lamb waves in metal sheets with experimental verification, 
Journal of Physics D: Applied Physics, 29 (1996) 147-155.
[13]F. Ricci, E. Monaco, N.D. Boffa, L. Maio, V. Memmolo, Guided waves for 
structural health monitoring in composites: A review and implementation strategies, 
Progress in Aerospace Sciences, 129 (2022) 100790.
[14]L. Moreau, A. Velichko, P.D. Wilcox, Accurate finite element modelling of guided 
wave scattering from irregular defects, Ndt & E International, 45 (2012) 46-54.
[15]D.T. Nguyen, L. Li, H. Ji, Stable and accurate numerical methods for generalized 
Kirchhoff–Love plates, Journal of Engineering Mathematics, 130 (2021) 6.
[16]E. Glushkov, N. Glushkova, A. Eremin, V. Giurgiutiu, Low-cost simulation of 
guided wave propagation in notched plate-like structures, Journal of Sound and 
Vibration, 352 (2015) 80-91.
[17]K. Harumi, F. Suzuki, Y. Satô, Computer simulation of the nearfield for elastic 
wave in a solid half‐space, The Journal of the Acoustical Society of America, 53 (1973) 
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 42

660-664.
[18]M. Alber, A. Buganza Tepole, W.R. Cannon, S. De, S. Dura-Bernal, K. Garikipati, 
G. Karniadakis, W.W. Lytton, P. Perdikaris, L. Petzold, Integrating machine learning 
and multiscale modeling—perspectives, challenges, and opportunities in the biological, 
biomedical, and behavioral sciences, NPJ digital medicine, 2 (2019) 115.
[19]B. Moseley, T. Nissen-Meyer, A. Markham, Deep learning for fast simulation of 
seismic waves in complex media, Solid earth, 11 (2020) 1527-1549.
[20]W. Shao, H. Sun, Q. Zhou, Y. Wang, X. Qing, A novel Lamb wave-based multi-
damage dataset construction and quantification algorithm under the framework of 
multi-task deep learning, Structural Health Monitoring, 23 (2024) 1148-1169.
[21]Z. Zhang, L. Wang, J. Li, Z. Zeng, Y. Liu, On acoustic fields of Lamb wave 
scattering in plates based on Convolutional Neural Network-Transformer, NDT & E 
International, (2025) 103628.
[22]X. Wang, J. Li, D. Wang, X. Huang, L. Liang, Z. Tang, Z. Fan, Y. Liu, Sparse 
ultrasonic guided wave imaging with compressive sensing and deep learning, 
Mechanical Systems and Signal Processing, 178 (2022) 109346.
[23]D. Wang, X. Wang, S. Chen, J. Li, L. Liang, Y. Liu, Joint learning of sparse and 
limited-view guided waves signals for feature reconstruction and imaging, Ultrasonics, 
137 (2024) 107200.
[24]L. Yang, X. Meng, G.E. Karniadakis, B-PINNs: Bayesian physics-informed neural 
networks for forward and inverse PDE problems with noisy data, Journal of 
Computational Physics, 425 (2021) 109913.
[25]S. Cuomo, V.S. Di Cola, F. Giampaolo, G. Rozza, M. Raissi, F. Piccialli, Scientific 
machine learning through physics–informed neural networks: Where we are and what’s 
next, Journal of Scientific Computing, 92 (2022) 88.
[26]A.I. Al-Adly, P. Kripakaran, Physics-informed neural networks for structural 
health monitoring: a case study for Kirchhoff–Love plates, Data-Centric Engineering, 
5 (2024) e6.
[27]M. Raissi, P. Perdikaris, G.E. Karniadakis, Physics-informed neural networks: A 
deep learning framework for solving forward and inverse problems involving nonlinear 
partial differential equations, Journal of Computational physics, 378 (2019) 686-707.
[28]M. Rasht‐Behesht, C. Huber, K. Shukla, G.E. Karniadakis, Physics‐informed 
neural networks (PINNs) for wave propagation and full waveform inversions, Journal 
of Geophysical Research: Solid Earth, 127 (2022) e2021JB023120.
[29]L. Lu, P. Jin, G. Pang, Z. Zhang, G.E. Karniadakis, Learning nonlinear operators 
via DeepONet based on the universal approximation theorem of operators, Nature 
machine intelligence, 3 (2021) 218-229.
[30]S. Wang, H. Wang, P. Perdikaris, Learning the solution operator of parametric 
partial differential equations with physics-informed DeepONets, Science advances, 7 
(2021) eabi8605.
[31]S. Nair, T.F. Walsh, G. Pickrell, F. Semperlotti, Physics and geometry informed 
neural operator network with application to acoustic scattering, Mechanical Systems 
and Signal Processing, 239 (2025) 113293.
[32]L. Wang, H. Wang, L. Liang, J. Li, Z. Zeng, Y. Liu, Physics-informed neural 
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed

## PDF页 43

networks for transcranial ultrasound wave propagation, Ultrasonics, 132 (2023) 107026.
[33]H. Wang, J. Li, L. Wang, L. Liang, Z. Zeng, Y. Liu, On acoustic fields of complex 
scatters based on physics-informed neural networks, Ultrasonics, 128 (2023) 106872.
[34]L. Wang, J. Li, S. Chen, Z. Fan, Z. Zeng, Y. Liu, Finite difference-embedded UNet 
for solving transcranial ultrasound frequency-domain wavefield, The Journal of the 
Acoustical Society of America, 155 (2024) 2257-2269.
[35]L. Wang, H. Zhang, Z. Zhang, H. Sun, X. Ying, J. Hou, J. Li, Z. Zeng, Y. Liu, S0 
Lamb wave scattering in plate structures using physics-enhanced TransUNet, 
Ultrasonics, (2025) 107852.
[36]Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. 
Anandkumar, Fourier neural operator for parametric partial differential equations, 
arXiv preprint arXiv:2010.08895, (2020).
[37]Z. Li, H. Zheng, N. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, A. 
Anandkumar, Physics-informed neural operator for learning partial differential 
equations, ACM/IMS Journal of Data Science, 1 (2024) 1-27.
[38]Z. Wang, X. Ying, J. Tong, W. Luo, F. Lv, Z. Tang, Y. Liu, Feature compensation 
and network reconstruction imaging with high-order helical modes in cylindrical 
waveguides, Ultrasonics, 151 (2025) 107631.
[39]X. Wang, M. Lin, J. Li, J. Tong, X. Huang, L. Liang, Z. Fan, Y. Liu, Ultrasonic 
guided wave imaging with deep learning: Applications in corrosion mapping, 
Mechanical Systems and Signal Processing, 169 (2022) 108761.
[40]L. Lv, S. Chen, J. Tong, X. Chen, Z. Zeng, Y. Liu, Ultrasonic guided wave imaging 
of pipelines based on physics embedded inversion neural network, Measurement 
Science and Technology, 34 (2023) 115401.
[41]S. Timoshenko, S. Woinowsky-Krieger, Theory of plates and shells, (1959).
[42]A.N. Norris, C. Vemula, Scattering of flexural waves on thin plates, Journal of 
sound and vibration, 181 (1995) 115-125.
This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=6967141
Preprint not peer reviewed