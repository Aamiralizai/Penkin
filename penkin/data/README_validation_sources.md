# External validation sources

## GSE9825
Public microarray series used for the high-versus-low producer expression ratios.

## Weber et al. 2012
DOI: 10.1128/AEM.01529-12

`weber2012_digitized.csv` contains approximate values digitized from published figures. Gene copy number is not equated to modeled catalytic-capacity scaling.

## Nijland et al. 2010
DOI: 10.1128/AEM.01702-10

`nijland2010_digitized.csv` contains approximate values read from published Figures 3B and 5. The analysis evaluates both copy-proportional scaling and a protein-informed scaling derived from the published protein-level plot.

## Janoska et al. 2022/2023
DOI: 10.1002/elsc.202100139

`janoska2022_oxygen_summary.csv` contains the reported dissolved-oxygen conditions and approximate relative penicillin-rate endpoints read from the published trajectory. ACV-increase and IPN-decrease directions are supported by the article text.

## Theilgaard et al. 2001
DOI: 10.1002/1097-0290(20000220)72:4<379::AID-BIT1000>3.0.CO;2-5

The abstract reports 124% and 176% increases for whole-cluster transformants and a 9% decrease for a pcbC-penDE transformant. Because exact enzyme-capacity folds for these transformants are not supplied in the abstract, the pipeline uses 2x and 3x capacity scenarios only as brackets and does not claim point-matched reconstruction.

## Douma et al. 2011 / GSE24212
DOI: 10.1186/1752-0509-5-132

The study reports a >10-fold decline in penicillin productivity, an approximately 3-fold decline in ACVS protein and a 5-20-fold decline in IPNS protein, while transcript changes were much smaller. The pipeline uses these reported protein changes as a stress test of the transcript-to-capacity approximation.

## Construction-based consistency sources

Deshmukh et al. 2015, DOI: 10.1016/j.ymben.2015.09.018

Douma et al. 2012, DOI: 10.1002/btpr.1503

These studies support pathway architecture, PAA response, branch behavior, and reversible PenG transport. Because these features informed model construction, they are not counted as independent validation in the external validation scorecard.
