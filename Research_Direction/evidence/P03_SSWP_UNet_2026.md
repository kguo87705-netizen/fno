# P03_SSWP_UNet_2026.pdf

## PDF页 1

A steady-state wavefield polarity-aware U-Net for ultrasonic guided waves 
depth-resolved imaging
Xuan Li
a
, Lishuai Liu
a , *
, Jiang Lu
a
, Yuncheng Zhang
a
, Dazhi Cong
b
, Yanxun Xiang
a , **
a
Shanghai Key Laboratory of Intelligent Sensing and Detection Technology, School of Mechanical and Power Engineering, East China University of Science and 
Technology, Shanghai, 200237, China
b
Shanghai Nuclear Engineering Research & Design Institute Co., LTD, Shanghai, 200233, China
ARTICLE INFO
Keywords:
Ultrasonic guided waves
Laser scanning full wavefield capture
Steady-state wavefield polarity
Depth-resolved imaging
ABSTRACT
The laser scanning full wavefield capture (LSFWC) technique enables rapid acquisition of two-dimensional maps 
of local velocity variations of ultrasonic guided waves (UGWs), thereby facilitating quantitative spatial visual -
ization of damage in thin-walled structures. However, this technique faces a trade-off among spatial resolution, 
the accuracy of damage estimation, and detection efficiency: the spatial sampling rate influences the limit of 
velocity estimation, whereas increasing the sampling rate renders the full wavefield data acquisition extremely 
time-consuming. To tackle these difficulties, we propose a U-Net based imaging approach to reconstruct high- 
quality depth-resolved images from steady-state wavefield polarity (SSWP) of spatial sub-Nyquist sampled 
UGWs. The developed SSWP-aware U-Net can automatically segment the wavefield, extract local phase infor -
mation, and perform encoding and decoding to establish approximate nonlinear mapping between wavefield and 
spatial distribution of UGWs velocity variations. The selection of SSWP as input features greatly accelerates the 
acquisition of the training dataset, and further enhances the robustness and generalization ability of U-Net 
trained on simulated dataset. We experimentally compared the proposed imaging method to wavenumber 
estimation methods. The results demonstrated that the proposed SSWP-aware U-Net is significantly superior to 
traditional methods in terms of image reconstruction quality and defect depth resolution especially with local 
spatial sub-Nyquist sampling.
1. Introduction
With the rapid advancement of modern industrial society, the de -
mand for the reliability and integrity assessment of materials, compo -
nents, and structures has been continuously increasing, thereby 
imposing higher requirements on nondestructive testing and evaluation 
(NDT & E) as well as structural health monitoring (SHM) technologies 
[ 1 – 3 ]. As a specialized modality of ultrasonic testing (UT), ultrasonic 
guided waves (UGWs) inspection not only retains the advantages of 
strong penetration, high sensitivity to volumetric defects, low cost, and 
safety for human operators, but also enables long-range propagation and 
the inspection of entire cross-section of waveguide structures through a 
single excitation, thereby demonstrating significant potential for engi -
neering applications and considerable research value [ 4 – 6 ]. Leveraging 
features such as amplitude variation [ 7 ], phase deviation [ 8 ], time of 
flight (ToF) [ 9 ], reflection and transmission ratios [ 10 ], energy 
attenuation [ 11 ], and mode conversion [ 12 ] of damage-scattered waves, 
UGWs locating/imaging techniques (including phased array [ 13 ], sparse 
array [ 14 ], and tomography [ 15 ]) have been developed. Nevertheless, 
accurately characterizing the severity of damage remains a significant 
challenge.
In contrast to contact-based UGWs transduction methods, laser 
scanning full wavefield capture (LSFWC) technique offers flexible and 
high-precision measurement of the full surface wavefield, enabling 
detailed observation of wavefield anomalies arising from damage-wave 
interactions [ 16 – 18 ]. This capability makes it an indispensable tool for 
in-depth investigations into the coupling mechanisms between struc -
tural damage and UGWs propagation. By exploiting the dispersive 
characteristics of UGWs, where the wavenumber of specific modes 
varies with the frequency-thickness product [ 19 ], depth-resolved 
detection of defects can be achieved based on laser-scanned full wave -
field measurements. A critical step toward this goal is the extraction of 
* Corresponding author.
** Corresponding author.
E-mail addresses: lishuai.liu@ecust.edu.cn (L. Liu), yxxiang@ecust.edu.cn (Y. Xiang). 
Contents lists available at ScienceDirect
NDT and E International
journal homep age: www. elsevier.com/ locate/ndte int
https://doi.org/10.1016/j.ndteint.2025.103597
Received 21 July 2025; Received in revised form 13 November 2025; Accepted 14 November 2025  
NDT&E International 158 (2026) 103597 
Available online 15 November 2025 
0963-8695/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies. 

## PDF页 2

the two-dimensional wavenumber field from the measured full wave -
field. Among the available approaches, Instantaneous Wavenumber 
Estimation (IWE) and Local Wavenumber Estimation (LWE) are 
currently the most widely adopted techniques for characterizing the 
spatial distribution of wavenumber associated with UGWs propagation 
[ 20 ].
The IWE algorithm estimates the local wavenumber by calculating 
the two-dimensional spatial phase gradient of the full wavefield 
[ 21–23 ]. Although computationally simple and efficient, making it 
suitable for real-time signal processing, it is sensitive to phase distor -
tions caused by wave interference, noise, or defect scattering. These 
distortions can compromise defect detection reliability and hinder the 
precise reconstruction of the wavenumber field. Consequently, the IWE 
method places greater demands on the Signal-to-Noise Ratio (SNR) and 
anti-interference of the laser vibrometer instrument as well as the sta -
bility of the measurement environment. To address these limitations, 
recent studies have proposed various strategies to improve the perfor -
mance of the IWE algorithm. These approaches, such as multi-shot [ 23 ] 
and multi-frequency [ 24 ] fusion method, have achieved notable im -
provements in noise robustness and defect detection accuracy, demon -
strating the potential for enhancing the reliability and applicability of 
IWE-based methods in practical structural health monitoring sce -
narios. While the LWE method extracts the local wavenumber by 
applying a three-dimensional Short-Space Fourier Transform (SSFT) to 
the full wavefield data [ 25–27 ]. The introduction of spatial window 
significantly enhances the robustness of the wavenumber spectrum 
against noise and mode interference. However, it also introduces a 
fundamental trade-off between spatial resolution and wavenumber 
resolution, making it difficult to achieve both high spatial and high 
wavenumber resolution simultaneously. In addition, the LWE method 
requires the spatial window to be moved point-by-point across the laser 
scanning area to compute the local wavenumber spectrum, leading to 
substantial computational workload and extended data processing 
times.
However, the laser scanning interval inherently determines the 
wavenumber range of all spatial wavenumber estimation algorithms. 
When the local UGWs velocity decreases to a critical threshold (the 
spatial sampling interval becomes larger than twice the wavelength), the 
local wavenumber extracted from the full wavefield data becomes dis -
torted. While reducing the scanning interval can yield a higher-quality 
spatial wavenumber spectrum, it undoubtedly decreases the efficiency 
of the LSFWC process. An alternative approach is to employ compressed 
sensing (CS) or deep learning super-resolution (DLSR) methods for the 
high-resolution reconstruction of full wavefield data with spatial sub- 
Nyquist sampling.
CS represents a signal or image in a sparse form using an appropriate 
basis, such as Fourier transform, cosine transform, or wavelet transform 
[ 28 , 29 ]. By exploiting sparsity and optimization methods for solving 
underdetermined equations, LSFWC data can be reconstructed at spatial 
sub-Nyquist sampling rates. Mesnil and Ruzzene [ 30 ] utilized a set of 
sparse measurement signals to identify the locations of excitation 
sources and scattering sources. Di Ianni et al. [ 31 ] investigated various 
basis functions in compressed sensing to reduce the time of LSFWC 
process. Wang et al. [ 32 ] designed a comprehensive dictionary incor -
porating a variety of waveforms generated under different conditions, 
enabling the decomposition of received signals in the spatial domain for 
damage localization, and employed a norm-based optimization algo -
rithm to obtain a sparse solution corresponding to the physical location 
of the damage. However, the accuracy of the reconstruction is signifi -
cantly influenced by the elements of the dictionary matrix, which is 
defined by the propagation equations of UGWs. Therefore, the acquisi -
tion of precise dispersion curves of UGWs in the structure is required a 
priori [ 33 ]. Additionally, factors such as the environmental tempera -
ture, and the potential interaction mechanisms between damage and 
UGWs can introduce uncertainty and distortion into the propagation 
equations [ 34 ]. Consequently, the dependence on the dictionary matrix 
and its sensitivity to uncontrollable factors may pose challenges to 
achieving robust and accurate wavefield reconstruction using CS 
algorithms.
Alternatively, deep learning offers an effective means to directly 
model the complex relationship between full guided wavefield and 
structural states by leveraging its automatic feature extraction and 
powerful nonlinear representation capabilities [ 35 , 36 ]. Ijjeh et al. [ 33 ] 
demonstrated DLSR method to reconstruct the high-resolution full 
wavefield from LSFWC data with spatial sub-Nyquist sampling. Their 
results indicated that DLSR outperforms CS in wavefield reconstruction 
accuracy, particularly in scenarios involving wave propagation through 
layered or heterogeneous regions. Ji et al. [ 17 , 34 ] successively proposed 
two deep learning architectures designed to accurately reconstruct the 
full wavefield and its spatial gradients from highly sparse measure -
ments. These studies showcased the capability of deep learning to 
effectively process sub-Nyquist sampled wavefield data and achieve 
high-resolution damage imaging. However, DLSR approaches also 
exhibit notable limitations. The performance of deep learning models is 
highly dependent on the quality and diversity of the training dataset. 
Acquiring a large number of experimental samples is often 
time-consuming and cost-prohibitive. Although simulation-based data 
generation offers an alternative, the transient time-domain computa -
tions required to obtain full wavefield data are computationally inten -
sive [ 37 ]. Furthermore, models trained solely on simulated dataset may 
suffer from limited robustness and generalizability when applied to 
LSFWC data, due to discrepancies between simulated and experimental 
conditions.
In this article, we propose a steady-state wavefield polarity (SSWP) 
-aware U-Net to reconstruct high-quality depth-resolved images from 
LSFWC data with spatial sub-Nyquist sampling. The SSWP-aware U-Net 
architecture consists of a symmetric encoder-decoder framework with 
skip connections and takes the SSWP of full guided wavefield as input, 
which enables to automatically model nonlinear interactions between 
the inspected medium and propagating UGWs, thus, establishing the 
mapping function linking the LSFWC data to the spatial distribution of 
the thickness within the inspected object. The SSWP of training dataset 
is directly generated by finite element analysis (FEA) in the frequency 
domain and the SSWP of LSFWC data is extracted from the direct 
wavefield, which is obtained through reference wavefield-based 
filtering. The performance of SSWP-aware U-Net is validated using 
LSFWC data from several aluminum alloy plate specimens with artificial 
thinning defects of various shapes, stepped and linear depth. The image 
quality of SSWP-aware U-Net is compared against the conventional 
wavenumber estimation methods to demonstrate the superior perfor -
mance of proposed SSWP-aware U-Net on the image reconstruction 
quality and defect depth resolution especially with local spatial sub- 
Nyquist sampling.
2. Methodology
2.1. SSWP extraction
In FEA, the steady-state solution of the full guided wavefield can be 
directly obtained through frequency-domain computation. However, to 
prevent the superposition of reflected and incident waves in the complex 
amplitude spectrum, perfectly matched layer (PML) must be introduced 
to absorb outward-propagating waves, thereby eliminating boundary 
reflections and minimizing their interference within the computational 
domain. In practical measurements, boundary reflections of closed 
structures are inherent due to the closed nature of the structure. To 
extract the steady-state wavefield from LSFWC data, it is also necessary 
to eliminate the boundary reflection waves. One approach involves the 
use of damping materials to create absorbing boundaries; however, this 
method is relatively complex to implement and requires careful selec -
tion of materials and boundary configurations tailored to different 
modes and frequencies of UGWs. A more flexible alternative is to 
X. Li et al.                                                                                                                                                                                                                                        NDT and E International 158 (2026) 103597 
2 

## PDF页 3

develop signal post-processing techniques based on filtering algorithms.
For a given structure, the theoretical guided wavefield can be 
computed using various analytical or numerical methods. This theoret -
ical wavefield serves as the basis for constructing a reference wavefield- 
based filter. In the case of a typical isotropic plate-like structure, the 
propagation of UGWs can be described by the Navier equation, which 
governs elastic wave motion in solid media 
( λ + μ )∇(∇ ⋅ u ) + μ ∇
2
u = ρ
∂
2
u
∂ t
2
(1) 
where the boundary conditions of UGWs in plate-like structure are σ
zz
=
σ
xz
= σ
yz
≡ 0 |
z =± h
, and 
u =
⎧
⎨
⎩
U
x
( z ) e
[ i ( k x x + k y y + k z z   ω t )]
U
y
( z ) e
[ i ( k x x + k y y + k z z   ω t )]
U
z
( z ) e
[ i ( k x x + k y y + k z z   ω t )]
(2) 
where 
{
U
x
, U
y
, U
z
}
are the amplitude distribution of the particle vibra -
tion in directions of x , y , z . ∇ is the Laplacian operator and λ and μ is the 
Lam ´e constants.
For full guided wavefield imaging, the direct wavefield inherently 
contains features indicative of structural damage. Therefore, reference 
wavefield filtering requires only the extraction of the direct wavefield 
component. To compute the reference wavefield v
ref
( x , y , t ) while satis -
fying the requirement of retaining only the direct wavefield component, 
the ray approximation method provides an efficient solution framework 
for the wave equation. This approach relies on two fundamental as -
sumptions: (1) guided waves propagate without boundary reflections, 
and (2) their local wavefronts exhibit plane-wave characteristics with 
energy confinement along predefined ray paths. Under these prescribed 
conditions, the reference wavefield can be formulated as: 
v
ref
( x , y , t ) = F
  1
{
F { s ( t )} e
  ik c ( f )
̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅ ̅̅̅̅̅̅̅̅ ̅
( x   x
0
)
2
+ ( y   y
0 )
2
√
}
(3) 
where F { ⋅ } and F
  1
{ ⋅ } are the Fourier transform and the inverse 
Fourier transform, 
 
x
0
, y
0
)
are the excitation position of UGWs, s ( t ) is the 
excitation signal, k
c
( f ) is the dispersion relation of detecting UGWs. The 
filter for isolating direct wavefield can be constructed by envelope of the 
reference wavefield 
W
R
( x , y , t ) =
⃒
⃒
v
ref
( x , y , t ) + i H
{
v
ref
( x , y , t )
} ⃒
⃒
(4) 
where the H { ⋅ } represents the Hilbert transform. By applying reference 
wavefield-based filter W
R
( x , y , t ) on the raw LSFWC data v ( x , y , t ) , the 
direct wavefield without boundary reflections can be obtained 
̃v ( x , y , t ) = v ( x , y , t ) W
R
( x , y , t ) (5) 
Then the direct wavefield in the spatio-temporal domain is trans -
formed into the frequency-wavenumber domain by Fourier transform 
and the steady-state wavefield of central frequency f
c 
can be extracted 
̃
V ( x , y , f
c
) = F { ̃v ( x , y , t )}|
f c
(6) 
Finally, the SSWP of UGWs is obtained by applying the sign function 
Sign { ⋅ } to the steady-state wavefield 
P ( x , y , f
c
) = Sign {
̃
V ( x , y , f
c
)} (7) 
where the notation Sign { ⋅ } denotes the sign function, which returns the 
sign of its argument (i.e., + 1 for positive values,   1 for negative values, 
and 0 if the value is zero).
By eliminating amplitude-related components from the original 
wavefield, the extraction of the polarization feature reduces suscepti -
bility to amplitude errors caused by environmental variations, surface 
Fig. 1. The procedures of laser scanning full wavefield capture and steady-state wavefield extraction.
X. Li et al.                                                                                                                                                                                                                                        NDT and E International 158 (2026) 103597 
3 

## PDF页 4

irregularities, and measurement uncertainties, thus enhancing the 
robustness of the network input. Fig. 1 shows the procedures of LSFWC 
and steady-state wavefield extraction. It is noted that the SSWP contains 
both real and imaginary components, which are utilized as two separate 
input channels for the deep learning model.
2.2. Proposed SSWP-aware U-Net
The SSWP can be regarded as a typical two-dimensional image. U- 
Net, a well-established deep learning architecture for image processing, 
is particularly suitable for SSWP based imaging task due to its symmetric 
encoder-decoder structure and concatenation-based skip connections. 
This design allows for the preservation of rich spatial details, enabling 
automatic segmentation of the SSWP, extraction of local phase infor -
mation, and the execution of encoding and decoding operations. 
Consequently, it facilitates the approximation of the nonlinear mapping 
between the guided wavefield and the spatial distribution of UGWs 
velocity variations. The architecture of proposed SSWP-aware U-Net is 
illustrated in Fig. 2 .
The proposed SSWP-aware U-Net architecture is a three-level 
encoder-decoder structure with skip connections whose input is the 
SSWP of UGWs and output is the generated depth-resolved image. The 
contracting path at the left side of the network structure can be regarded 
as an encoder to perform feature extraction to capture the spatial phase 
Fig. 2. The proposed SSWP-aware U-Net architecture.
Fig. 3. Schematic diagrams of finite element simulation model.
X. Li et al.                                                                                                                                                                                                                                        NDT and E International 158 (2026) 103597 
4 

## PDF页 5

distribution information in the SSWP data. The symmetric expanding 
path at the right side can be regarded as a decoder to generate the 
corresponding depth-resolved image. The size of input SSWP data is 
100 × 100 × 2, where 100 is the number of spatial sampling points of 
LSFWC in the directions of x and y , and 2 denotes the two input channels 
corresponding to the real and imaginary parts of the SSWP, respectively. 
Each encoding layer contains two 3 × 3 convolution kernels, each of 
which goes through a two-dimensional batch normalization layer and a 
leaky rectified linear unit (LeakyReLU), enabling the network to capture 
nonlinear features effectively and mitigate the risk of neuron inactiva -
tion in the negative input domain. Additionally, a dropout rate of 0 . 1 is 
applied after the activation function to prevent overfitting. Each 
encoding layer is then followed by a 2 × 2 max pooling layer for 
downsampling. The input SSWP data go through three downsampling 
layers, so the spatial dimension of the data tensor is contracted from 
100 × 100 to 12 × 12. The decoding layer is similar to the encoding 
layer, but the max pooling is replaced by a 2 × 2 transposed convolution 
with a stride of 2, which gradually restores the resolution of the feature 
maps through the upsampling operation to reconstruct the depth- 
resolved image. In the final output layer, an additional upsampling 
operation is introduced using a 2 × 2 transposed convolution with a 
stride of 2, which increases the spatial resolution of the output to 200 ×
200, thereby enhancing the imaging resolution. It is worth noting that, 
since the input to the SSWP-aware U-Net does not conform to standard 
image dimensions, dimensional mismatches may arise during the 
downsampling and upsampling operations, affecting spatially aligned 
concatenation. To address this, a tensor padding operation was applied 
after the first upsampling stage, expanding the feature map from 24 × 24 
to 25 × 25. This ensures consistent spatial dimensions between corre -
sponding layers in the encoder and decoder, preserving the structural 
symmetry of the network.
3. Implementation procedures
3.1. Dataset preparation
This section aims to demonstrate the training of a deep learning 
model capable of transforming LSFWC data into depth-resolved images. 
A supervised learning approach is employed to train the proposed SSWP- 
aware U-Net architecture, establishing a mapping between the SSWP of 
LSFWC data and the two-dimensional spatial distribution of the thick -
ness of the inspected structure. However, obtaining large-scale experi -
mental datasets that simultaneously contain LSFWC data and ground- 
truth thickness maps is both impractical and cost-prohibitive. 
Additionally, generating such datasets via time-domain FEA demands 
substantial computational resources and time. Therefore, the steady- 
state wavefield derived from the LSFWC data is adopted as the input 
feature for the network, as it can be efficiently obtained through 
frequency-domain FEA. To enhance the robustness and generalization 
capability of the U-Net trained on simulated dataset, the steady-state 
wavefield is further processed into the SSWP, which serves as the 
input feature to the network. The polarization operation removes 
amplitude-related information from the original wavefield, thereby 
mitigating errors in amplitude that may arise from environmental fac -
tors, surface roughness, or other measurement uncertainties in practical 
scenarios.
The steady-state wavefields of UGWs used for training data are 
generated using the solid mechanics module and frequency domain 
analysis in COMSOL Multiphysics 6.2. Fig. 3 shows the schematic dia -
grams of finite element simulation model. The simulation model consists 
of a 340 mm × 340 mm × 2 mm aluminum plate with a density of 
2700 kg / m
3
, Young ’ s modulus of 70 GPa , and a Poisson ’ s ratio of 0 . 33, 
and no material damping. PMLs are implemented along all boundaries to 
absorb outgoing waves and eliminate reflections. Each PML has a width 
of 20 mm , with a scaling factor of 1.3 and a scaling curvature parameter 
of 3. The central 200 mm × 200 mm region of the plate includes a series of 
randomly distributed defects with varying sizes and depths, which are 
generated using MATLAB. These defects are designed to simulate real -
istic thinning-type damage and serve as the target features within the 
imaging region. The A
0 
mode of the UGWs was excited by applying point 
loads with identical orientation on both the upper and lower surfaces at 
( 318 , 170 ) . For frequency-domain FEA, the guided waves can be 
generated using a single time-independent continuous force, with the 
effective excitation frequency determined by the frequency range 
computed by the solver. The simulation model is meshed using 
quadratic serendipity elements with shape of free tetrahedral and 
element size of 0 . 5 mm . The steady-state wavefield was computed using 
frequency-domain FEA with an implicit solver, with the PARDISO direct 
solver applied at 400 kHz to ensure numerical stability. The steady-state 
wavefields used for training are sampled at interval of 2 mm , yielding 
input data of size 100 × 100, whereas the ground-truth depth-resolved 
images are sampled at interval of 2 mm , resulting in label maps of size 
200 × 200. The simulated dataset contains 6000 ground-truth depth- 
resolved images, along with their corresponding steady-state wavefield 
data. Each frequency-domain simulation requires approximately 160 
seconds of computation on a platform configured with 288 GB RAM, 
Intel Xeon Silver 4210R CPU @ 2.40 GHz and NVIDIA RTX A6000 GPU. 
The total computational time for generating the entire dataset is 
Fig. 4. Illustration of the bottom and side surfaces of the specimens (unit: mm).
X. Li et al.                                                                                                                                                                                                                                        NDT and E International 158 (2026) 103597 
5 

## PDF页 6

approximately 270 hours, which is substantially lower than that 
required for constructing time-domain simulated dataset of the same 
size [ 37 ].
3.2. Training strategy
The training of the proposed SSWP-aware U-Net is first conducted on 
a simulated dataset and subsequently validated on an experimental 
dataset. In the frequency-domain FEA, a total of 6000 SSWP samples are 
generated and used as inputs to the network, while the corresponding 
depth-resolved maps served as the target outputs. The entire set of 
simulated samples was randomly split into training and testing subsets 
with a ratio of 9:1, where the training set was used for model optimi -
zation and the testing set for performance evaluation. Final performance 
assessment of the trained network was carried out using the experi -
mental LSFWC dataset. During training, the adaptive moment estima -
tion with weight decay (AdamW) optimizer is employed with an initial 
learning rate of 0 . 001 and weight decay coefficient of 0 . 0001 to mini -
mize the loss function. To prevent overfitting, a dropout strategy was 
adopted, and batch normalization was incorporated to mitigate the ef -
fects of parameter initialization and accelerate convergence. To 
compare the network performance, a U-Net model with the same ar -
chitecture was also trained using the steady-state wavefield as input, 
under identical training strategy. All training procedures were per -
formed on the NVIDIA RTX A6000 GPU.
3.3. Experimental setup
To validate the generalizability of proposed SSWP-aware U-Net on 
real UGWs data, experimental LSFWC data are acquired from aluminum 
plate specimens containing a series of artificially machined thinning 
defects with various shapes, stepped cross sections, and linear depth 
variations. The details about the bottom and side surfaces of the speci -
mens were illustrated in Fig. 4 . A schematic of the experimental setup is 
presented in Fig. 5 . As shown in Fig. 5 , UGWs are excited using a PZT-5H 
transducer with dimensions of 5 mm × 5 mm × 5 mm . The excitation signal 
consisted of a 5-cycle tone burst modulated by a Hanning window with a 
central frequency of 400 kHz . This signal was generated by Tektronix 
AFG31022 arbitrary waveform generator and subsequently amplified 
using Agitek ATA-2022B high-voltage amplifier. LSFWC data are ac -
quired using Sound & Bright Tempo 2D two-wave mixing laser interfer -
ometric vibrometer. The specimen was mounted on an x-y scanning 
platform driven by two sets of stepper motors. The scanning platform 
was synchronously controlled via the trigger signal from the arbitrary 
waveform generator to enable precise spatial acquisition of the full 
wavefield. The temporal sampling rate is 25 MHz , and the spatial sam -
pling interval is set to 2 mm .
Fig. 5. Experiment configuration.
Fig. 6. Average loss of the network on the simulated dataset (a) SSWP-aware U-Net; (b) steady-state wavefield-aware U-Net.
X. Li et al.                                                                                                                                                                                                                                        NDT and E International 158 (2026) 103597 
6 

## PDF页 7

Fig. 7. (a – c) Real thickness distribution map of the specimens; Depth-resolved imaging results by (d – f) proposed SSWP-aware U-Net, (g – i) steady-state wavefield- 
aware U-Net, (j – l) LWE and (m – o) IWE. The dashed lines represent the boundaries of the defects.
X. Li et al.                                                                                                                                                                                                                                        NDT and E International 158 (2026) 103597 
7 

## PDF页 8

4. Results and discussion
As illustrated in Fig. 6 , the average training and testing loss of SSWP- 
aware U-Net and steady-state wavefield-aware U-Net on the simulated 
dataset is monitored to evaluate model convergence and training sta -
bility. The loss curves indicate that both networks effectively learned the 
correlation between the wavefield and the spatial distribution of defect 
depth. Specifically, the SSWP-aware U-Net exhibits slight overfitting, as 
evidenced by a noticeable gap between the training and testing losses; 
however, both remained within the same order of magnitude. The 
steady-state wavefield-aware U-Net achieves marginally lower overall 
losses than the SSWP-aware U-Net, though the difference was not sub -
stantial, and the losses also remain within the same order of magnitude. 
The actual performance and generalization capability of the networks 
are further validated and compared against traditional wavenumber 
estimation-based imaging methods using experimental LSFWC data.
Fig. 7 shows the actual thickness distribution map of the specimens 
along with the imaging results obtained using the proposed SSWP-aware 
U-Net and other depth-resolved imaging methods, including steady- 
state wavefield-aware U-Net, LWE and IWE. It can be observed that 
the proposed SSWP-aware U-Net reconstructed images successfully 
capture the fine morphological features of the defects, with noise and 
artifacts effectively suppressed. This enables accurate assessment of 
both the shape and depth of each defect. These experimental results 
confirm the high-resolution depth-resolved imaging capability of the 
proposed deep learning-based approach. While the steady-state wave -
field-aware U-Net with the same network architecture is also capable of 
accurately characterizing defects, its imaging results contain a consid -
erable number of artifact regions. This can be attributed to the domain 
gap between the simulated and experimental data. In experimental 
conditions, factors such as environmental noise and variations in surface 
reflectivity of the specimen introduce additional disturbances to the 
amplitude distribution of the full wavefield. By leveraging only phase- 
related features through the SSWP input, the SSWP-aware U-Net is 
less sensitive to amplitude distortions caused by experimental un -
certainties, thereby achieving more consistent and reliable imaging 
performance when transitioning from simulated to real LSFWC data. In 
traditional wavenumber estimation-based imaging methods, significant 
imaging noise is commonly observed. Moreover, according to the sam -
pling theorem, the fixed spatial sampling interval of 2 mm in the 
experimental setup limits the maximum achievable depth resolution to 
approximately 1 . 2 mm . As a result, these methods tend to produce 
substantial depth estimation errors, particularly in regions containing 
deeper defects. To provide a clearer evaluation of the depth-resolved 
imaging performance, Fig. 8 presents depth profile cross-sections 
extracted along the vertical centerlines of two specimens with 
different thinning depth. The results demonstrate that the proposed 
SSWP-aware U-Net yields a significantly higher imaging signal-to-noise 
ratio compared to traditional methods and maintains reliable depth 
estimation performance even in areas with sub-Nyquist sampling, 
particularly in regions where the defect depth exceeds 1 . 2 mm . In 
summary, experimental results demonstrate that the proposed SSWP- 
aware U-Net exhibits strong cross-domain robustness and remarkably 
outperforms traditional wavenumber estimation methods in recon -
structing high-quality depth-resolved images. In particular, it shows 
superior performance in artifact and noise suppression, as well as in 
overcoming the resolution limitations imposed by spatial sampling 
constraints.
5. Conclusion
In this work, we propose a deep learning-based approach aimed at 
directly reconstructing high-quality depth-resolved images from the 
SSWP of full guided wavefield data. Specifically, based on the two- 
dimensional UGWs propagation model, we design a reference 
wavefield-based filter that enables the extraction of boundary- 
reflection-free SSWP from LSFWC data. To replace conventional wave -
number estimation imaging algorithms, we develop an end-to-end 
SSWP-aware U-Net. Benefiting from the encoder-decoder architecture 
with skip connections, the proposed deep learning model is capable of 
implicitly capturing the complex nonlinear interactions between struc -
tural features and UGWs propagation, thereby learning an effective 
mapping between the spatial phase distribution of the steady-state 
wavefield and the spatial thickness distribution of the inspected struc -
ture. By using SSWP, which removes amplitude information and retains 
only phase-related features, as the input feature, the network achieves 
more consistent and robust imaging performance when transitioning 
from simulated data to real LSFWC data. Experimental results validate 
that the proposed SSWP-aware U-Net significantly outperforms tradi -
tional wavenumber estimation methods in reconstructing high- 
resolution depth-resolved images, particularly in terms of artifact and 
noise suppression and in overcoming resolution limits imposed by 
spatial sampling constraints. The proposed approach holds promise for a 
wide range of laser-scanned guided waves imaging applications.
CRediT authorship contribution statement
Xuan Li: Writing – original draft, Methodology, Conceptualization. 
Lishuai Liu: Writing – review & editing, Investigation, Funding acqui -
sition. Jiang Lu: Visualization, Software, Methodology. Yuncheng 
Zhang: Supervision, Formal analysis, Data curation. Dazhi Cong: 
Fig. 8. Profiles from depth-resolved imaging results of specimens containing 
artificially machined thinning defects with (a) stepped cross sections and (b) 
linear depth variations, at the center of the vertical direction.
X. Li et al.                                                                                                                                                                                                                                        NDT and E International 158 (2026) 103597 
8 

## PDF页 9

Formal analysis, Data curation. Yanxun Xiang: Project administration, 
Funding acquisition.
Declaration of competing interest
The authors declare that they have no known competing financial 
interests or personal relationships that could have appeared to influence 
the work reported in this paper.
Acknowledgment
This work was supported by the National Key Research and Devel -
opment Program of China (Grant No. 2024YFF0619000), the National 
Natural Science Foundation of China (Grant Nos. 12422415, 12327807 
and 12374434) and the Basic Research Program of Shanghai Science and 
Technology Innovation Action Plan (No. No. 24TS1412200 and 
24ZR1490900).
Data availability
Data will be made available on request.
References
[1] Lang YF, Yang ZB, Kong DT, Zhang W, Chen XF. Forward-propagation-free focusing 
MUSIC algorithm for Lamb waves. Struct Health Monit 2023;22(6):1–15 .
[2] Hua J, Gao F, Li W. Local wavenumber imaging with low-sampled wavefield for 
delamination characterization in composite plate. NDT&E Int 2023;137:102867 .
[3] Wang J, Shen Y. An enhanced Lamb wave virtual time reversal technique for 
damage detection with transducer transfer function compensation. Smart Mater 
Struct 2019;28(8):085017 .
[4] Shen Y, Wang J, Xu W. Nonlinear features of guided wave scattering from rivet 
hole nucleated fatigue cracks considering the rough contact surface condition. 
Smart Mater Struct 2018;27(10):105044 .
[5] Liu YX, Sun KH, Yang TX, Zhang JQ, Ta D, Li D. A robust lamb wave imaging 
approach to plate-like structural health monitoring of materials with transducer 
array position errors. IEEE T Ultrason Ferr 2022;69(6):2162–77 .
[6] Lang YF, Yang ZB, Yang LH, Chen XF. Lamb wave frequency diverse array. IEEE T 
Ultrason Ferr 2022;69(8):2526–39 .
[7] Ling FY, Chen HL, Lang YF, Yang ZB, Xu KL, Ta D. Lamb wave tomography for 
defect localization using wideband dispersion reversal method. Measurement 
2023;216:112965 .
[8] Rajagopalan J, Multila K, Khmoa, Balasubramaniam K, Krishnamurthy CV. A phase 
reconstruction algorithm for Lamb wave based structural health monitoring of 
anisotropic multilayered composite plates. J Acoust Soc Am 2006;119(2):872–8 .
[9] Clarke T, Cawley P, Wilcox PD, Croxford AJ. Evaluation of the damage detection 
capability of a sparse-array guided-wave SHM system applied to a complex 
structure under varying thermal conditions. IEEE T Ultrason Ferr 2009;56(12): 
2666–78 .
[10] Ostachowicz W, Kudela P, Malinowski P, Wandowski T. Damage localisation in 
plate-like structures based on PZT sensors. Mech Syst Signal Process 2009;23(6): 
1805–29 .
[11] Wu ZJ, Qing XLP, Chang F-K. Damage detection for composite laminate plates with 
a distributed hybrid PZT/FBG sensor network. J Intel Mat Syst Str 2009;20(9): 
1069–77 .
[12] Zhang J, Drinkwater BW, Wilcox PD, Hunter AJ. Defect detection using ultrasonic 
arrays: the multi-mode total focusing method. NDT&E Int 2010;43(2):123–33 .
[13] Li X, Liu L, Xu H, Hu Z, Xiang Y, Xuan F-Z. Lamb wave phased array imaging based 
on phase-amplitude compounding algorithm. Mech Syst Signal Process 2023;205 
(15):110882 .
[14] Yang B, Xuan F-Z, Chen S, Zhou S, Gao Y, Xiao B. Damage localization and 
identification in WGF/epoxy composite laminates by using Lamb waves: 
experiment and simulation. Compos Struct 2017;165(1):138–47 .
[15] Zhao X, Gao H, Zhang G, Ayhan B, Yan F, Kwan C, Rose JL. Active health 
monitoring of an aircraft wing with embedded piezoelectric sensor/actuator 
network: I. Defect detection, localization and growth monitoring. Struct Health 
Monit 2007;16(4):1208–17 .
[16] Sun Z, Wu S, Saini A, Fan Z. Characterizing polycrystalline microstructures by 
reconstructing grain boundaries and velocity map from ultrasonic surface wave 
field. Ultrasonics 2024;137:107175 .
[17] Ji D, Lin J, Gao F, Hua J, Li W. A deep learning-based spatial gradient 
reconstruction method for efficient damage identification in composite with high- 
sparsity Lamb wavefield. Mech Syst Signal Process 2025;224(1):112018 .
[18] Harley JB, Chia CC. Statistical partial wavefield imaging using Lamb wave signals. 
Struct Health Monit 2017;17(4):1–17 .
[19] Rose JL. Ultrasonic guided waves in solid media. Cambridge University Press; 
2014 .
[20] Zheng S, Luo Y, Xu C, Xu G. A review of Laser ultrasonic Lamb wave damage 
detection methods for thin-walled structures. Sensors (Basel) 2023;23(6):3183 .
[21] Mesnil C A Leckey, Ruzzene M. Instantaneous and local wavenumber estimations 
for damage quantification in composites. Struct Health Monit 2015;14(3):193–204 .
[22] Mesnil, Yan H, Ruzzene M, Paynabar K, Shi JJ. Fast wavenumber measurement for 
accurate and automatic location and quantification of defect in composite. Struct 
Health Monit 2016;15(2):223–34 .
[23] Liu ZH, Liu XY, Tian JZ, Gao D, He CF, Wu B. Quantitative evaluation of corrosion- 
thinning defect by an instantaneous wavenumber multi-shot fusion method based 
on laser ultrasonic guided wavefield detection. Ultrasonics 2025;148:107561 .
[24] Liu XY, Liu ZH, Zhu YP, Lu ZJ, Chen L, Wu B, He CF. Multi-frequency instantaneous 
wavenumber damage imaging method based on ultrasonic guided waves induced 
by laser point-by-point excitation. Nondestruct Test Eva 2024;40(5):2208–24 .
[25] Rogge MD, Leckey CAC. Characterization of impact damage in composite laminates 
using guided wavefield imaging and local wavenumber domain analysis. 
Ultrasonics 2013;53(7):1217–26 .
[26] Spytek J, Pieczonka L. Local wavenumber imaging in anisotropic structures. Mech 
Syst Signal Process 2025;224(1):112102 .
[27] Flynn EB, Chong SY, Jarmer GJ, Lee J-R. Structural imaging through local 
wavenumber estimation of guided waves. NDT&E Int 2013;59:1–10 .
[28] Candes EJ, Romberg J, Tao T. Robust uncertainty principles: exact signal 
reconstruction from highly incomplete frequency information. IEEE T Inform 
Theory 2006;52(2):489–509 .
[29] Liu XY, Liu ZH, Zhu YP, Chen L, Tang ZH, He CF. Sparse multi-time wavenumber 
analysis method based on incomplete guided wavefield data. Appl Acoust 2025; 
241(5):111028 .
[30] Mesnil M Ruzzene. Sparse wavefield reconstruction and source detection using 
Compressed Sensing. Ultrasonics 2016;67:94–104 .
[31] Perelli L De Marchi, Flamigni L, Marzani A, Masetti G. Best basis compressive 
sensing of guided waves in structural health monitoring. Digit Signal Process: Rev J 
2015;42:35–42 .
[32] Wang W, Bao Y, Zhou W, Li H. Sparse representation for lamb-wave-based damage 
detection using a dictionary algorithm. Ultrasonics 2018;87:48–58 .
[33] Ijjeh S Ullah, Radzienski M, Kudela P. Deep learning super-resolution for the 
reconstruction of full wavefield of Lamb waves. Mech Syst Signal Process 2023;186 
(1):109878 .
[34] Ji D, Lin J, Gao F. The high-sparsity Lamb Wavefield Reconstruction framework 
based on masked autoencoder. IEEE T Instrum Meas 2023;72:4507310 .
[35] Hong X, Cai Y, Chen Z, Huang R, Zhang B. Deep knowledge transfer powered 
ultrasonic guided wave damage monitoring under incomplete database scenarios: 
theories, applications and challenges. Smart Mater Struct 2025;34:033003 .
[36] Liu L, Liu W, Teng D, Xiang Y, Xuan F-Z. A multiscale residual U-net architecture 
for super-resolution ultrasonic phased array imaging from full matrix capture data. 
J Acoust Soc Am 2023;154(4):2044–54 .
[37] Ijjeh, Ullah S, Kudela P. Full wavefield processing by using FCN for delamination 
detection. Mech Syst Signal Process 2021;153:107537 .
X. Li et al.                                                                                                                                                                                                                                        NDT and E International 158 (2026) 103597 
9 