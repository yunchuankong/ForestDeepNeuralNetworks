setwd("...")

detectNA <- function(row){
  if (sum(is.na(row))>=1){
    return(1)
  } else {
    return(0)
  }
}
manyZeros<-function(x){
  zeroRate=0.1
  if (length(which(x==min(x)))/length(x)>zeroRate)
    # if (length(which(x==0))/length(x)>zeroRate)
    return(1)
  else
    return(0)
}

z_score <- function(vec){
  (vec - mean(vec)) / sd(vec)
}

## check normalized data
par(mfrow=c(3,3))
for (i in 1:9){
  # pca <- princomp(X[,sample(ncol(X),900)])
  # v <- pca$scores[,1:2]
  # plot(v[y==1,],pch=16)
  # points(v[y==0,],col='red',pch=16)
  hist(X[,sample(ncol(X),1)],50)
}
par(mfrow=c(1,1))






