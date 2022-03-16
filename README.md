# iCREPCP
iCEEPCP is a deep learning-based web server to learn the cis-regulatory code and to identify cis-regulatory elements(CREs) of a given Plant Core Promoter.<br>
**Citation**: Kaixuan Deng#, Qizhe Zhang#, Yuxin Hong, Jianbing Yan* and Xuehai Hu*(2022), iCREPCP: a deep learning-based web server for identifying critical cis-regulatory elements within plant core promoters.<br>
### Requires
- tensorflow==2.4 (for model training)
- tensorflow==1.14, keras==2.3.1, deeplift==0.6.13.0 (for deepLIFT and TF-moDISco)
- pandas
- matplotlib
- seaborn
### Install
```
git clone git@https://github.com/kaixuanDeng95/iCREPCP.git
```
### A Short Intrduction of Our Work
<p>
Cis-regulatory modules(CRMs) and trans-acting factors(TAFs)  play an important role in specifying the quantitative level of gene expressions in plant biology.Common CRMs include gene-proximal promoters, distal enhancers, and silencers, which are all considered as the complex assemblies of cis-regulatory elements (CREs). The identification and characterization of plant CRMs or critical CREs will not only provide helpful knowledge for our understandings of transcriptional regulatory mechanisms in plants, but also is an essential prerequisite for plant breeding by genome editing.</p>
<p>
 Our focus is on Plant Core Promoters (PCP), a large group of CRMs, which can drive basal level of target gene transcriptions because they are rich in CREs (Nat Plants 2021). A typical PCP is the minimal sequence region (usually 50-100bp around transcription start site (TSS) of plant genes) needed to direct initiation of transcription ((Nat Plants 2022). The promoter strength of PCP is defined as the ability to drive expression of a barcoded green fluorescent protein (GFP) reporter gene in tobacco leaves or maize protoplasts systems (Nat Plants 2021).
 </p>
 <p>
  Here, we first trained a deep learning model to learn the cis-regulatory code of PCP and then developed a deep learning-based web server to identify which CREs a given Plant Core Promoter (iCREPCP) contains, with a focus on the exact position of each CRE and its contribution to the promoter strength. To achieve the above purpose, we first employed a large-scale dataset of 18,329 Arabidopsis, 34,415 maize and 27,094 sorghum core promoters, whose strengths were measured by STARR-seq assays via two transient transfection systems of tobacco leaves and maize protoplasts (ref Nat Plants 2021). We then trained a deep learning model of ‘DenseNet’ (Ref DenseNet) to accurately fit promoter strengths with their DNA sequences. Importantly, a powerful model interpretability tool of ‘DeepLIFT’ (Ref DeepLIFT) was employed to assign feature importance (via base contribution score) to each base in view of its contribution to make prediction of the promoter strength with the trained model. Finally, several successive bases with significantly high values of base contribution score were identified as critical CREs within a given PCP by ‘TF-MoDISco’ (Transcription Factor Motif Discovery from Importance Scores) (Ref TF-MoDISco). 
  </p>
