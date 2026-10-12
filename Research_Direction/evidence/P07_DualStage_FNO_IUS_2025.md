# P07_DualStage_FNO_IUS_2025.pdf

## PDF页 1

Dual-Stage Fourier Neural Operator: Mitigating 
Spectral Bias for Enhanced Ultrasonic Guided 
Wave Corrosion Imaging 
 
Yizhe Gao 
State Key Laboratory of Precision 
Measuring Technology and 
Instruments, Tianjin University 
Tianjin, China 
2023433025@tju.edu.cn 
Jian Li 
State Key Laboratory of Precision 
Measuring Technology and 
Instruments, Tianjin University 
Tianjin, China 
tjupipe@tju.edu.cn 
Linfeng Wang 
State Key Laboratory of Precision 
Measuring Technology and 
Instruments, Tianjin University 
Tianjin, China 
wanglinfeng@tju.edu.cn 
Yang Liu 
State Key Laboratory of Precision 
Measuring Technology and 
Instruments, Tianjin University 
Tianjin, China 
ultrasonicslab@tju.edu.cn 
Yitian Yan 
State Key Laboratory of Precision 
Measuring Technology and 
Instruments, Tianjin University 
Tianjin, China 
2019205372@tju.edu.cn 
Abstract—Deep neural networks exhibit spectral bias in  
data-driven ultrasonic guided wave corrosion imaging, 
preferentially learning low -frequency components while 
struggling with high -frequency features critical for defect 
boundary delineation. This paper proposes a Dual -Stage 
Fourier Neural Operator (DS -FNO) framework  designed to 
mitigate this limitation. The first stage recovers dominant low -
frequency velocity components, while the second stage employs 
residual learning to capture fine -scale spectral content. 
Numerical experiments on aluminum plate structures validate 
the feasibility and effectiveness of the proposed dual -stage 
architecture for enhanced corrosion imaging. 
Keywords—Fourier Neural Operator, spectral bias, 
ultrasonic guided wave, corrosion imaging 
I. INTRODUCTION  
Material degradation through corrosion presents 
significant economic and environmental challenges, with 
annual costs reaching substantial levels globally [1]. Among 
various non -destructive evaluation techniques for corrosion 
detection, ultrasonic guided wave (UGW) imaging has 
emerged as a particularly effective approach due to its 
capability for long- range inspection and sensitivity to 
thickness variations [2]. 
Recent advances in deep learning have revolutionized 
UGW-based defect reconstruction by establishing direct 
mappings between acoustic signals and damage 
characteristics [3 ,4]. While deep neural networks (DNNs) 
have demonstrated promising results in corrosion imaging 
tasks, they suffer from an inherent limitation known as 
spectral bias or the frequency principle (F -Principle). This 
phenomenon manifests as DNNs preferentially learning low-
frequency components while struggling to capture high -
frequency features during training [ 5,6]. In the context of 
defect imaging, this bias often leads to over -smoothed 
reconstructions that fail to preserve sharp boundaries and 
fine-scale features critical for accurate damage assessment. 
The spectral bias in conventional DNNs may arise from 
the smoothness properties of gradient -based optimization 
dynamics [ 7]. During training, networks  converge to low -
frequency solutions first, requiring more iterations to learn 
high-frequency components — if they can be learned at all. 
While this characteristic may contribute to generalization in 
many applications , it becomes particularly problematic for 
UGW imaging where sharp discontinuities at defect 
boundaries and fine structural details require accurate high-
frequency reconstruction. 
Recently, a new class of networks known as Neural 
Operators (NOs) has emerged as powerful surrogate models 
for solving Partial Differential Equations (PDEs). NOs can 
learn mappings between infinite -dimensional function 
spaces, with the Fourier Neural Operator (FNO) serving as a 
prominent representative [8]. However, even FNOs exhibit 
the spectral bias phenomenon, struggling with high-
frequency features essential for capturing multiscale details 
[9]. 
In this work, we propose a Dual -Stage Fourier Neural 
Operator (DS -FNO) framework specifically designed to 
mitigate the spectral bias limitation in data -driven UGW 
corrosion imaging. Our method employs a two- stage end-to-
end imaging approach that achieves high -quality corrosion 
reconstruction by incorporating a complementary second 
stage to compensate for the inherent frequency bias. The 
FNO serves as the primary decoder in this process, 
leveraging its function-space mapping capability to learn the 
highly nonlinear relationship between input acoustic features 
and velocity distributions. Section II introduces the forward 
modeling framework and the DS -FNO architecture. Section 
III discusses the numerical experiment configuration and 
inversion results. Section IV concludes the paper. 
II. METHODOLOGY 
This section presents the theoretical foundation and 
implementation details of the proposed DS -FNO framework 
for ultrasonic guided wave corrosion imaging. The 
methodology comprises two key components: first, a forward 
modeling approach that simulates wave propagation in 
damaged plates to generate comprehensive training datasets, 
and second, a dual-stage inversion architecture built upon the 
FNO. 
A. Forward Modeling 
To generate training data for corrosion characterization, 
we employ numerical simulation of guided wave propagation 
2025 IEEE International Ultrasonics Symposium (IUS) | 979-8-3315-2332-9/25/$31.00 ©2025 IEEE | DOI: 10.1109/IUS62464.2025.11201843
Authorized licensed use limited to: Harbin Institute of Technology. Downloaded on September 10,2026 at 10:59:59 UTC from IEEE Xplore.  Restrictions apply. 

## PDF页 2

in damaged plate structures. In the 2D space -time domain, 
the governing elastic wave equations for isotropic materials 
can be formulated as [10]: 
 
.. .
( ) ( ,) ( ) ( ,) ( ) ( ,) ( ,)t t tt++=M xux Cxux Sxux F x  (1) 
where ()Mx , ()Cx , and ()Sx represent the global mass, 
damping, and stiffness matrices respectively. The 
displacement field is denoted by ( ,) tux , 
while ( ,) tFx indicates the external excitation source. 
Transformation to the frequency domain yields [10]: 
  2( )(, ) (, )iωω ω ω− ++ =M C Sux F x  (2) 
where ω represents the angular frequency, 
and (, )ωux and (, )ωFx are the Fourier transforms of the 
temporal displacement and source fields respectively. This 
can be simplified by introducing the complex impedance 
matrix: 
 
 (, )(, ) (, )ωω ω =Zx ux F x  (3) 
where 2(, ) iωω ω= − ++Zx M C S represents the 
frequency-dependent impedance matrix. 
The frequency-domain formulation with finite difference 
discretization provides significant computational advantages, 
particularly through LU decomposition of the complex 
impedance matrix, which needs to be performed only once 
for multiple source configurations [11]. This approach 
substantially reduces the computational burden for 
generating the comprehensive dataset required for training. 
The relationship between the wavefield distribution 
u and the velocity model c , which is implicit in the 
aforementioned equations, can be characterized through a 
nonlinear forward operator: 
  ()=uc   (4) 
where  denotes the forward modeling operator 
and c represents the spatial velocity distribution. This 
nonlinear mapping forms the basis for our inverse problem 
formulation. 
To enable quantitative assessment, we exploit the 
relationship between velocity and structural thickness. The 
phase velocity  
pc of guided waves at a specific frequency 
exhibits a dispersive relationship with the plate thickness  h . 
This relationship, derived from the Rayleigh -Lamb 
frequency equations for the selected guided wave mode, can 
be expressed as:  
 ( ,) f= phc   (5) 
where  denotes the dispersion- based transformation 
operator and f  represents the excitation frequency. This 
mapping enables direct conversion between velocity maps 
and thickness distributions, facilitating quantitative corrosion 
assessment. 
 
Fig. 1. Fourier Neural Operator architecture with spectral convolution 
layers. 
B. Dual-Stage Inversion Design 
The objective of this study is to develop an operator 
learning framework that establishes robust mappings from 
acoustic signals to velocity models while mitigating inherent 
spectral bias. Our proposed DS -FNO approach employs 
FNO as the decoder in a two -stage process: the first stage 
primarily recovers low-frequency components of the velocity 
models, while the second stage specifically targets the fine -
scale spectral content that remains under -resolved in the 
initial reconstruction. By combining outputs from both 
stages, DS-FNO effectively recovers a broader spectrum of 
the inverted models, resulting in enhanced reconstruction 
accuracy. 
The FNO architecture, illustrated in Fig. 1 , leverages the 
convolution theorem in Fourier space to efficiently learn 
operator mappings. The architecture consists primarily of 
multiple Fourier layers that perform spectral convolutions. 
Each FNO layer can be expressed as:  
 
( )1 () () ( ) ()l ll l lσ+ = + v x Wv x v x  (6) 
where lv denotes the l -th layer features, σ is a nonlinear 
activation function, lW represents linear transformations, 
and l is the kernel integral operator parameterized in Fourier 
space [8]: 
 ( )
1( ) () ( ) ()ll l l
−= ⋅ vx R v x  (7) 
Here,  denotes the Fourier transform, and lR is a 
learnable complex-valued weight tensor that operates on the  
predetermined lowest Fourier modes, enabling feature 
mixing and transformation in the spectral domain. 
The DS-FNO framework architecture is illustrated in Fig. 
2, where both the Stage 1 and Stage 2 models employ four 
FNO layers with identical architectural configurations as 
their primary decoders. The two- stage reconstruction 
workflow proceeds as follows: 
The first stage network 
1 maps the input acoustic 
signals a to an initial velocity reconstruction 1c : 
 1 11(; )= ca θ  (8) 
where 1θ represents the trainable parameters of the first 
stage. Due to the inherent spectral characteristics of the FNO 
architecture, this initial reconstruction naturally emphasizes 
Authorized licensed use limited to: Harbin Institute of Technology. Downloaded on September 10,2026 at 10:59:59 UTC from IEEE Xplore.  Restrictions apply. 

## PDF页 3

the dominant low -frequency components of the velocity 
distribution. 
 
Fig. 2. Dual-stage FNO framework architecture.  
The second stage addresses the reconstruction errors by 
learning a complementary mapping. We define the residual 
as: 
 1= − r cc  (9) 
where r represents the residual between the true velocity 
map c and the Stage 1 prediction , serving as the target for 
Stage 2 training.  The Stage 2 network  2 takes both the 
original signals and the Stage 1 output as inputs: 
 122([ , ]; )= r ac θ  (10) 
where 2θ represents the trainable parameters of the 
second stage, and  r denotes the predicted residual 
correction. The concatenation  1[, ] ac provides the second 
stage with both the original signals and the current 
reconstruction state, enabling focused learning on the under -
resolved spectral components . To enhance the information 
flow between stages, we incorporate a learnable forward 
network that transforms the Stage 1 output before 
concatenation, allowing adaptive feature extraction for 
residual prediction. 
The final velocity map 
finalc combines both stage outputs: 
 final 1= + c cr  (11) 
III. NUMERICAL VALIDATION AND RESULTS 
A. Experimental Configuration 
 
Fig. 3. Experimental setup with aluminum plate and circular transducer 
array configuration. 
 
Fig. 4. Dispersion curves for A0 mode in aluminum plate. 
The numerical experiments were conducted on an 
aluminum plate model to validate the proposed DS -FNO 
framework for corrosion imaging. The simulation setup, 
illustrated in Fig. 3, consists of a 1000 mm × 1000 mm × 10 
mm aluminum plate with material properties: density 2.7 
g/cm³, Young's modulus 70.8 GPa, and Poisson's ratio 0.33.  
A circular array of 32 transducers with a diameter of 0.7 m 
was uniformly distributed to provide comprehensive 
wavefield coverage. Various irregular -shaped corrosion 
defects ranging from 0 to 3 mm in depth were randomly 
positioned within the transducer array to generate diverse 
training scenarios. The computational domain was 
discretized into a 250 × 250 grid with 4 mm spacing. For 
excitation, a 50 kHz Ricker wavelet was employed at each 
transducer location sequentially, while the remaining 
transducers recorded the scattered wavefields . The 
frequency-domain forward modeling directly provided 
complex-valued wavefield data, from which both real and 
imaginary components were extracted as network inputs, 
resulting in a 2 × 32 × 32 dimensional input tensor for each 
sample. The corresponding velocity maps served as ground 
truth labels for supervised training. 
Fig. 4 presents the dispersion curves calculated for the 
aluminum plate with the aforementioned material properties. 
The A0 mode, when excited at 50 kHz, exhibits strong 
sensitivity to thickness variations, making it suitable for 
corrosion depth characterization. Fig . 5 illustrates a 
representative example of the thickness -to-velocity 
conversion, demonstrating the monotonic relationship 
between corrosion depth and phase velocity that enables 
quantitative damage assessment through the dispersion-based 
transformation described in (5). 
 
Fig. 5. Thickness-to-velocity conversion example. 
B. Results Analysis 
The DS-FNO framework was trained and evaluated on a 
comprehensive dataset comprising 2000 samples generated 
through the forward modeling process described above. The 
dataset was partitioned into training, validation, and test sets 
Authorized licensed use limited to: Harbin Institute of Technology. Downloaded on September 10,2026 at 10:59:59 UTC from IEEE Xplore.  Restrictions apply. 

## PDF页 4

with a ratio of 1400:300:300, ensuring sufficient diversity for 
model generalization. Importantly, the sample allocation 
remained consistent across both training stages to maintain 
fair comparison and prevent data leakage between stages. 
 
Fig. 6. Inversion results for a representative test sample. 
Both stages of the DS -FNO were trained with identical 
hyperparameters, employing the same optimizer and 
normalized mean squared error (MSE) loss function. Each 
stage was trained with early stopping based on validation 
loss to prevent overfitting. The first stage was trained 
independently to convergence, after which its weights were 
frozen during the second stage training to focus exclusively 
on residual learning. 
Fig. 6 presents a reconstruction example from the test set, 
demonstrating the progressive improvement achieved 
through the dual -stage architecture. Fig. 6(a) shows the 
ground truth thickness map with an irregular corrosion 
defect, while Fig.  6(b) displays the final combined output 
from the DS -FNO framework, showing excellent agreement 
with the ground truth. Fig. 6(c) provides a quantitative cross-
sectional comparison at a randomly selected location (x=540 
mm), where the combined output (dashed line) closely 
follows the ground truth profile (solid line), accurately 
capturing both the defect depth and its spatial extent. Fig. 
6(d) presents the Stage 1 output, which successfully 
identifies the defect location but exhibits over -smoothed 
boundaries. Fig. 6(e) shows the actual residual between the 
ground truth and Stage 1 output, revealing the high -
frequency components that need to be recovered. Figs. 6(f) 
and 6(g) display the Stage 2 output and its corresponding 
residual, demonstrating how the second stage compensates 
for the high -frequency components missed in the initial 
reconstruction, effectively capturing fine -scale features at 
defect boundaries and validating the dual-stage approach. 
IV. C
ONCLUSION 
This work presents a Dual-Stage Fourier Neural Operator 
framework designed to address the spectral bias inherent in 
DNN-based ultrasonic guided wave corrosion imaging. By 
employing a two-stage architecture, the method first utilizes 
FNO to capture dominant low -frequency components, then 
applies residual learning to recover high- frequency details. 
Experimental validation demonstrate s the framework's 
ability to achieve high-quality corrosion reconstruction. 
ACKNOWLEDGMENT 
This work was supported in part by National Science 
Foundation of China under Contract No. 12474459, in part 
by the Natural Science Foundation of Zhejiang Province 
under Contract No. LY23E010001. 
REFERENCES 
[1] B. Hou et al. , “The cost of corrosion in China,” npj Materials 
Degradation, vol. 1, no. 1, p. 4, 2017. 
[2] J. L. Rose, Ultrasonic Guided Waves in Solid Media , Cambridge 
University Press, 2014. 
[3] X. Wang et al., “Ultrasonic guided wave imaging with deep learning: 
Applications in corrosion mapping,” Mechanical Systems and Signal 
Processing, vol. 169, p. 108761, 2022. 
[4] D. Wang, X. Wang, S. Chen, J. Li, L. Liang, and Y. Liu, “Joint 
learning of sparse and limited -view guided waves signals for feature 
reconstruction and imaging,” Ultrasonics, vol. 137, p. 107200, 2024. 
[5] N. Rahaman et al., “On the Spectral Bias of Neural Networks ,” 
Proceedings of  International Conference on Machine Learning, pp.  
5301–5310, 2019. 
[6] Z.-Q. J. Xu, Y. Zhang, T. Luo, Y. Xiao, and Z. Ma, “Frequency 
Principle: Fourier Analysis Sheds Light on Deep Neural Networks,” 
Communications in Computational Physics, vol. 28, no. 5, pp. 1746 –
1767, 2020. 
[7] B. Ronen, D. Jacobs, Y. Kasten, and S. Kritchman, “The 
Convergence Rate of Neural Networks for Learned Functions of 
Different Frequencies ,” Proceedings of Advances in Neural 
Information Processing Systems, vol. 32, 2019. 
[8] Z. Li et al., “Fourier Neural Operator for Parametric Partial 
Differential Equations,” arXiv:2010.08895, 2020. 
[9] X. Liu, B. Xu, S. Cao, and L. Zhang, “Mitigating spectral bias for the 
multiscale operator learning,” Journal of Computational Physics, vol. 
506, p. 112944, 2024. 
[10] K. J. Marfurt, “Accuracy of finite‐difference and finite‐element 
modeling of the scalar and elastic wave equations,” Geophysics , vol. 
49, no. 5, pp. 533–549, 1984. 
[11] M. Lin and Y. Liu, “Guided Wave Tomography Based on Supervised 
Descent Method for Quantitative Corrosion Imaging,” IEEE 
transactions on ultrasonics, ferroelectrics, and frequency control, vol. 
68, no. 12, pp. 3624–3636, 2021. 
 
Authorized licensed use limited to: Harbin Institute of Technology. Downloaded on September 10,2026 at 10:59:59 UTC from IEEE Xplore.  Restrictions apply. 