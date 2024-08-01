Deep Ensemble Encoder Network (DEEN)

Overview
Genome-wide association studies (GWAS) have identified numerous single nucleotide polymorphisms (SNPs) associated with various heritable human traits and diseases. Traditional polygenic risk scores (PRS) capture only simple linear genetic effects across the genome. To address this limitation, we introduce Deep Ensemble Encoder Network (DEEN), which combines autoencoders and fully connected neural networks (FCNNs) to better identify and model both linear and non-linear SNP effects across different genomic regions, improving disease risk prediction.

Usage
The script getPRSresult.py generates PRS results for your data.

Parameters:

--disease: Specifies the disease under the Data folder. Your data should be located under ./RawFiles/DiseaseName/, with raw data for all 22 chromosomes under this path (e.g., ./RawFiles/Hypertension/Chr1.raw).
--pheno: Place your phenotype file under the ./Data/PhenoFiles folder.
--save: If you wish to save the data as a NumPy file, specify this parameter. Without it, the data will not be saved.

Sample Usage
python ./getPRSresult.py --disease Hypertension --pheno ./Data/PhenoFiles/Hypertension --save outputResult.npy
python ./getPRSresult.py --disease T2D --pheno ./Data/PhenoFiles/T2D --save outputResult.npy

Data Preparation
1. Raw Data: Ensure all 22 chromosome raw data files are placed under the appropriate disease folder within ./RawFiles/DiseaseName/.
2. Phenotype Files: Place your phenotype files in the ./Data/PhenoFiles folder.

Note: The sample data is created with: 

values = ["0", "1", "2", "NA"]
probabilities = [0.6, 0.25, 0.13, 0.02]

You can download Hypertension models from

