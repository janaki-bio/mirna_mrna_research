# Exports the PAM50 subtype calls (TCGAquery_subtype("BRCA")) and the tumour purity estimates (Tumor.purity, BRCA rows) from
# the data bundled with TCGAbiolinks, and the package versions.
# Writes external/tcgabiolinks/brca_subtype.csv, tumor_purity_brca.csv and versions.txt (paths relative to the working directory).
# Run from analysis/ in the R environment (environment-r.yml):  Rscript 01_export_pam50_purity.R
suppressMessages(library(TCGAbiolinks))
out <- "external/tcgabiolinks"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
sub <- TCGAquery_subtype("BRCA")
write.csv(sub, file.path(out, "brca_subtype.csv"), row.names = FALSE)
pur <- get("Tumor.purity")
pur <- pur[pur$Cancer.type == "BRCA", ]
write.csv(pur, file.path(out, "tumor_purity_brca.csv"), row.names = FALSE)
writeLines(c(paste("TCGAbiolinks", packageVersion("TCGAbiolinks")), R.version.string,
             paste("subtype_rows", nrow(sub)), paste("purity_rows", nrow(pur))), file.path(out, "versions.txt"))
