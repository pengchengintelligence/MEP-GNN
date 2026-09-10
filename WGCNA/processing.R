library(WGCNA)
library(flashClust)
library(openxlsx)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
input_csv <- ifelse(length(args) >= 1, args[1], Sys.getenv("MEP_GNN_WGCNA_INPUT", "/data/ROSMAP/mRNA.csv"))
output_adj <- ifelse(length(args) >= 2, args[2], Sys.getenv("MEP_GNN_WGCNA_OUTPUT_ADJ", "/data/ROSMAP/mRNA_adj.csv"))
label_csv <- ifelse(length(args) >= 3, args[3], Sys.getenv("MEP_GNN_WGCNA_LABELS", "data/ROSMAP/labels_tr.csv"))

d <- read.csv(input_csv, stringsAsFactors = FALSE, header = TRUE, check.names = FALSE)
expression.data <- as.data.frame(d)
expression.data <- as.data.frame(lapply(expression.data, as.numeric)) 

gsg <-goodSamplesGenes(expression.data)
summary(gsg)
gsg$allOK

if (!gsg$allOK)
{
  if (sum(!gsg$goodGenes)>0) 
    printFlush(paste("Removing genes:", paste(names(expression.data)[!gsg$goodGenes], collapse = ", "))); 
  if (sum(!gsg$goodSamples)>0)
    printFlush(paste("Removing samples:", paste(rownames(expression.data)[!gsg$goodSamples], collapse = ", "))); 
  expression.data <- expression.data[gsg$goodSamples == TRUE, gsg$goodGenes == TRUE] 
}

sampleTree <- hclust(dist(expression.data), method = "average") 


par(cex = 0.6);
par(mar = c(0,4,2,0))

plot(sampleTree, main = "Sample clustering to detect outliers", sub="", xlab="", cex.lab = 1.5,
     cex.axis = 1.5, cex.main = 2)


spt <- pickSoftThreshold(expression.data) 
spt

par(mar=c(1,1,1,1))
plot(spt$fitIndices[,1],spt$fitIndices[,2],
     xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit,signed R^2",type="n",
     main = paste("Scale independence"))
text(spt$fitIndices[,1],spt$fitIndices[,2],col="red")
abline(h=0.80,col="red")

par(mar=c(1,1,1,1))
plot(spt$fitIndices[,1], spt$fitIndices[,5],
     xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
     main = paste("Mean connectivity"))
text(spt$fitIndices[,1], spt$fitIndices[,5], labels= spt$fitIndices[,1],col="red")

softPower <- spt$fitIndices$Power[which.max(spt$fitIndices$SFT.R.sq > 0.8)]
adjacency <- adjacency(expression.data, power = softPower)

write.table(
  adjacency,
  file = output_adj,
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

TOM <- TOMsimilarity(adjacency)
TOM.dissimilarity <- 1-TOM

geneTree <- hclust(as.dist(TOM.dissimilarity), method = "average") 

sizeGrWindow(12,9)
plot(geneTree, xlab="", sub="", main = "Gene clustering on TOM-based dissimilarity", 
     labels = FALSE, hang = 0.04)

Modules <- cutreeDynamic(dendro = geneTree, distM = TOM.dissimilarity, deepSplit = 2, 
                         pamRespectsDendro = FALSE, minClusterSize = 4)

table(Modules)
module_table <- as.data.frame(table(Modules))

ModuleColors <- labels2colors(Modules) 
table(ModuleColors) 
module_color_table <- as.data.frame(table(ModuleColors))

plotDendroAndColors(geneTree, ModuleColors,"Module",
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05,
                    main = "Gene dendrogram and module colors")

dev.off()

MElist <- moduleEigengenes(expression.data, colors = ModuleColors) 
MEs <- MElist$eigengenes 

label.data <- read.csv(label_csv, header = FALSE)

colnames(label.data) <- c("Label")

rownames(label.data) <- rownames(expression.data)

label.data$Label <- as.numeric(as.character(label.data$Label))

moduleTraitCor <- cor(MEs, label.data, use = "p")
moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nrow(expression.data))

textMatrix <- paste(signif(moduleTraitCor, 2), "\n(",
                    signif(moduleTraitPvalue, 1), ")", sep = "")
dim(textMatrix) <- dim(moduleTraitCor)

labeledHeatmap(Matrix = moduleTraitCor,
               xLabels = colnames(label.data),
               yLabels = names(MEs),
               ySymbols = names(MEs),
               colorLabels = FALSE,
               colors = blueWhiteRed(50),
               textMatrix = textMatrix,
               setStdMargins = FALSE,
               cex.text = 0.5,
               zlim = c(-1,1),
               main = "Module-trait relationships")






