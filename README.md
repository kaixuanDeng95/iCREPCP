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
### Intrduction
<p>
Cis-regulatory modules(CRMs) and trans-acting factors(TAFs)  play an important role in specifying the quantitative level of gene expressions in plant biology.Common CRMs include gene-proximal promoters, distal enhancers, and silencers, which are all considered as the complex assemblies of cis-regulatory elements (CREs). The identification and characterization of plant CRMs or critical CREs will not only provide helpful knowledge for our understandings of transcriptional regulatory mechanisms in plants, but also is an essential prerequisite for plant breeding by genome editing. </p>
<p>
